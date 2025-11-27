use anyhow::{anyhow, Result};
use bytemuck::{cast_slice, try_cast_slice};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};

use crate::sensitivity::allowed;

#[derive(Clone)]
pub struct RagStore {
    path: PathBuf,
}

impl RagStore {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let store = Self {
            path: path.as_ref().to_path_buf(),
        };
        store.init()?;
        Ok(store)
    }

    fn connection(&self) -> Result<Connection> {
        Ok(Connection::open(&self.path)?)
    }

    pub fn init(&self) -> Result<()> {
        let conn = self.connection()?;
        conn.execute_batch(
            r#"
            PRAGMA journal_mode = WAL;
            CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                collection_id INTEGER NOT NULL,
                source_path TEXT NOT NULL,
                dcf_path TEXT,
                title TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(collection_id) REFERENCES collections(id)
            );
            CREATE TABLE IF NOT EXISTS cells (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page INTEGER NOT NULL,
                importance INTEGER NOT NULL,
                sensitivity TEXT NOT NULL DEFAULT 'public',
                text TEXT,
                text_encrypted BLOB,
                encryption TEXT,
                embedding BLOB NOT NULL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );
            CREATE INDEX IF NOT EXISTS idx_documents_collection ON documents(collection_id);
            CREATE INDEX IF NOT EXISTS idx_cells_document ON cells(document_id);
            "#,
        )?;
        Ok(())
    }

    pub fn ensure_collection(&self, name: &str) -> Result<i64> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare("SELECT id FROM collections WHERE name = ?1")?;
        if let Some(id) = stmt.query_row([name], |row| row.get(0)).optional()? {
            return Ok(id);
        }
        conn.execute("INSERT INTO collections (name) VALUES (?1)", params![name])?;
        Ok(conn.last_insert_rowid())
    }

    pub fn add_document(&self, collection_id: i64, doc: &DocumentInsert) -> Result<DocumentRecord> {
        let conn = self.connection()?;
        conn.execute(
            "INSERT INTO documents (collection_id, source_path, dcf_path, title) VALUES (?1, ?2, ?3, ?4)",
            params![collection_id, doc.source_path, doc.dcf_path, doc.title],
        )?;
        let id = conn.last_insert_rowid();
        Ok(DocumentRecord {
            id,
            source_path: doc.source_path.clone(),
        })
    }

    pub fn add_cells(&self, document_id: i64, cells: &[CellInsert]) -> Result<usize> {
        let mut conn = self.connection()?;
        let tx = conn.transaction()?;
        for cell in cells {
            let embedding_blob = cast_slice::<f32, u8>(&cell.embedding);
            tx.execute(
                "INSERT INTO cells (document_id, page, importance, sensitivity, text, text_encrypted, encryption, embedding, bbox_x, bbox_y, bbox_w, bbox_h) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                rusqlite::params![
                    document_id,
                    cell.page,
                    cell.importance as i64,
                    cell.sensitivity,
                    cell.text,
                    cell.text_encrypted,
                    cell.encryption,
                    embedding_blob,
                    cell.bbox_x,
                    cell.bbox_y,
                    cell.bbox_w,
                    cell.bbox_h
                ],
            )?;
        }
        tx.commit()?;
        Ok(cells.len())
    }

    pub fn search_cells(
        &self,
        collection: &str,
        query_embedding: &[f32],
        filters: &SearchFilters,
    ) -> Result<Vec<ScoredCell>> {
        let conn = self.connection()?;
        let mut stmt = conn.prepare(
            r#"
            SELECT
                cells.id,
                documents.id,
                documents.source_path,
                cells.page,
                cells.importance,
                cells.sensitivity,
                cells.text,
                cells.text_encrypted,
                cells.encryption,
                cells.embedding,
                cells.bbox_x,
                cells.bbox_y,
                cells.bbox_w,
                cells.bbox_h
            FROM cells
            JOIN documents ON cells.document_id = documents.id
            JOIN collections ON documents.collection_id = collections.id
            WHERE collections.name = ?1
            "#,
        )?;
        let mut rows = stmt.query([collection])?;
        let mut hits = Vec::new();
        while let Some(row) = rows.next()? {
            let doc_id: i64 = row.get(1)?;
            let source_path: String = row.get(2)?;
            let sensitivity: String = row.get(5)?;
            if !allowed(&sensitivity, &filters.sensitivity_threshold) {
                continue;
            }
            let text_encrypted: Option<Vec<u8>> = row.get(7)?;
            if filters.policy == RagPolicy::External && text_encrypted.is_some() {
                continue;
            }
            let encryption: Option<String> = row.get(8)?;
            let embedding_blob: Vec<u8> = row.get(9)?;
            let embedding: &[f32] =
                try_cast_slice(&embedding_blob).map_err(|_| anyhow!("invalid embedding"))?;
            let score = cosine_similarity(query_embedding, embedding);
            hits.push(ScoredCell {
                cell_id: row.get(0)?,
                document_id: doc_id,
                document_source: source_path,
                page: row.get(3)?,
                importance: row.get::<_, i64>(4)? as u8,
                sensitivity,
                text: row.get(6)?,
                text_encrypted,
                encryption,
                bbox_x: row.get(10)?,
                bbox_y: row.get(11)?,
                bbox_w: row.get(12)?,
                bbox_h: row.get(13)?,
                score,
            });
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if hits.len() > filters.top_k {
            hits.truncate(filters.top_k);
        }
        Ok(hits)
    }
}

#[derive(Debug, Clone)]
pub struct DocumentInsert {
    pub source_path: String,
    pub dcf_path: Option<String>,
    pub title: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DocumentRecord {
    pub id: i64,
    pub source_path: String,
}

#[derive(Debug, Clone)]
pub struct CellInsert {
    pub page: u32,
    pub importance: u8,
    pub sensitivity: String,
    pub text: Option<String>,
    pub text_encrypted: Option<Vec<u8>>,
    pub encryption: Option<String>,
    pub embedding: Vec<f32>,
    pub bbox_x: i32,
    pub bbox_y: i32,
    pub bbox_w: u32,
    pub bbox_h: u32,
}

#[derive(Debug, Clone)]
pub struct SearchFilters {
    pub top_k: usize,
    pub sensitivity_threshold: String,
    pub policy: RagPolicy,
}

impl Default for SearchFilters {
    fn default() -> Self {
        Self {
            top_k: 10,
            sensitivity_threshold: "public".to_string(),
            policy: RagPolicy::External,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ScoredCell {
    pub cell_id: i64,
    pub document_id: i64,
    pub document_source: String,
    pub page: i64,
    pub importance: u8,
    pub sensitivity: String,
    pub text: Option<String>,
    pub text_encrypted: Option<Vec<u8>>,
    pub encryption: Option<String>,
    pub bbox_x: i32,
    pub bbox_y: i32,
    pub bbox_w: u32,
    pub bbox_h: u32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagPolicy {
    External,
    Internal,
}

impl Default for RagPolicy {
    fn default() -> Self {
        RagPolicy::External
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }
    dot / (a_norm.sqrt() * b_norm.sqrt())
}
