use std::collections::HashSet;
use std::convert::TryFrom;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use indexmap::IndexMap;
use prost::Message;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::proto;

pub type CodeHash = [u8; 32];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub version: u32,
    pub grid: String,
    pub codeset: String,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            version: 1,
            grid: "coarse".to_string(),
            codeset: "HASH256".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CellType {
    Text,
    Table,
    Figure,
    Footer,
    Header,
}

impl From<CellType> for proto::CellType {
    fn from(value: CellType) -> Self {
        match value {
            CellType::Text => proto::CellType::Text,
            CellType::Table => proto::CellType::Table,
            CellType::Figure => proto::CellType::Figure,
            CellType::Footer => proto::CellType::Footer,
            CellType::Header => proto::CellType::Header,
        }
    }
}

impl From<proto::CellType> for CellType {
    fn from(value: proto::CellType) -> Self {
        match value {
            proto::CellType::Text => CellType::Text,
            proto::CellType::Table => CellType::Table,
            proto::CellType::Figure => CellType::Figure,
            proto::CellType::Footer => CellType::Footer,
            proto::CellType::Header => CellType::Header,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PageInfo {
    pub z: u32,
    pub width_px: u32,
    pub height_px: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CellRecord {
    pub z: u32,
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
    #[serde(with = "codehash_serde")]
    pub code_id: CodeHash,
    pub rle: u32,
    pub cell_type: CellType,
    pub importance: u8,
}

impl CellRecord {
    pub fn key(&self) -> (u32, i32, i32) {
        (self.z, self.y, self.x)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct NumGuard {
    pub z: u32,
    pub x: u32,
    pub y: u32,
    pub units: String,
    #[serde(with = "numhash_serde")]
    pub sha1: [u8; 20],
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Document {
    pub header: Header,
    pub pages: Vec<PageInfo>,
    pub cells: Vec<CellRecord>,
    #[serde(with = "dict_serde")]
    pub dict: IndexMap<CodeHash, String>,
    pub numguards: Vec<NumGuard>,
}

impl Document {
    pub fn new(header: Header) -> Self {
        Self {
            header,
            pages: Vec::new(),
            cells: Vec::new(),
            dict: IndexMap::new(),
            numguards: Vec::new(),
        }
    }

    pub fn add_page(&mut self, info: PageInfo) {
        self.pages.push(info);
    }

    pub fn push_cell(&mut self, cell: CellRecord, payload: String) {
        let code = cell.code_id;
        self.cells.push(cell);
        self.dict.entry(code).or_insert(payload);
    }

    pub fn add_numguard(&mut self, guard: NumGuard) {
        self.numguards.push(guard);
    }

    pub fn payload_for(&self, code_id: &CodeHash) -> Option<&str> {
        self.dict.get(code_id).map(|s| s.as_str())
    }

    pub fn ordered_cells(&self) -> Vec<CellRecord> {
        let mut cells = self.cells.clone();
        cells.sort_by_key(|c| (c.z, c.y, c.x));
        cells
    }

    pub fn to_proto(&self) -> proto::Document {
        let mut prev = (0i64, 0i64, 0i64);
        let cells = self
            .ordered_cells()
            .into_iter()
            .map(|cell| {
                let dz = cell.z as i64 - prev.0;
                let dx = cell.x as i64 - prev.1;
                let dy = cell.y as i64 - prev.2;
                prev = (cell.z as i64, cell.x as i64, cell.y as i64);
                proto::Cell {
                    dz: dz as i32,
                    dx: dx as i32,
                    dy: dy as i32,
                    w: cell.w,
                    h: cell.h,
                    code_id: cell.code_id.to_vec().into(),
                    rle: cell.rle,
                    r#type: proto::CellType::from(cell.cell_type) as i32,
                    importance_q: cell.importance as u32,
                }
            })
            .collect();

        let dict = self
            .dict
            .iter()
            .map(|(code_id, payload)| proto::DictEntry {
                code_id: code_id.to_vec().into(),
                payload_utf8: payload.clone(),
            })
            .collect();

        let numguards = self
            .numguards
            .iter()
            .map(|guard| proto::NumGuard {
                z: guard.z,
                x: guard.x,
                y: guard.y,
                units: guard.units.clone(),
                sha1: guard.sha1.to_vec().into(),
            })
            .collect();

        proto::Document {
            header: Some(proto::Header {
                version: self.header.version,
                grid: self.header.grid.clone(),
                codeset: self.header.codeset.clone(),
            }),
            pages: self
                .pages
                .iter()
                .map(|p| proto::PageInfo {
                    z: p.z,
                    width_px: p.width_px,
                    height_px: p.height_px,
                })
                .collect(),
            cells,
            dict,
            numguards,
        }
    }

    pub fn from_proto(doc: proto::Document) -> Result<Self> {
        let header = doc
            .header
            .map(|h| Header {
                version: h.version,
                grid: h.grid,
                codeset: h.codeset,
            })
            .unwrap_or_default();

        let pages = doc
            .pages
            .into_iter()
            .map(|p| PageInfo {
                z: p.z,
                width_px: p.width_px,
                height_px: p.height_px,
            })
            .collect();

        let mut cells = Vec::new();
        let mut prev = (0i64, 0i64, 0i64);
        for cell in doc.cells {
            prev.0 += cell.dz as i64;
            prev.1 += cell.dx as i64;
            prev.2 += cell.dy as i64;
            let mut code_id = [0u8; 32];
            code_id.copy_from_slice(&cell.code_id);
            cells.push(CellRecord {
                z: prev.0 as u32,
                x: prev.1 as i32,
                y: prev.2 as i32,
                w: cell.w,
                h: cell.h,
                code_id,
                rle: cell.rle,
                cell_type: proto::CellType::try_from(cell.r#type)
                    .map(CellType::from)
                    .unwrap_or(CellType::Text),
                importance: cell.importance_q as u8,
            });
        }

        let mut dict = IndexMap::new();
        for entry in doc.dict {
            let mut code_id = [0u8; 32];
            code_id.copy_from_slice(&entry.code_id);
            dict.insert(code_id, entry.payload_utf8);
        }

        let numguards = doc
            .numguards
            .into_iter()
            .map(|guard| {
                let mut sha = [0u8; 20];
                sha.copy_from_slice(&guard.sha1);
                NumGuard {
                    z: guard.z,
                    x: guard.x,
                    y: guard.y,
                    units: guard.units,
                    sha1: sha,
                }
            })
            .collect();

        Ok(Self {
            header,
            pages,
            cells,
            dict,
            numguards,
        })
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let proto = self.to_proto();
        let mut buf = Vec::with_capacity(proto.encoded_len());
        proto.encode(&mut buf)?;
        let mut encoder = zstd::stream::Encoder::new(Vec::new(), 3)?;
        encoder.write_all(&buf)?;
        let data = encoder.finish()?;
        Ok(data)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut decoder = zstd::stream::Decoder::new(bytes)?;
        let mut buf = Vec::new();
        decoder.read_to_end(&mut buf)?;
        let proto = proto::Document::decode(&*buf)?;
        Self::from_proto(proto)
    }

    pub fn save_bin<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let bytes = self.to_bytes()?;
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    pub fn load_bin<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        Self::from_bytes(&buf)
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let mut file = File::create(path)?;
        serde_json::to_writer_pretty(&mut file, self)?;
        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let doc: Document = serde_json::from_reader(file)?;
        Ok(doc)
    }

    pub fn total_cells(&self) -> usize {
        self.cells.len()
    }

    pub fn total_pages(&self) -> usize {
        self.pages.len()
    }

    pub fn ensure_dict_entry(&mut self, payload: &str) -> CodeHash {
        let hash = hash_payload(payload);
        self.dict.entry(hash).or_insert_with(|| payload.to_string());
        hash
    }

    pub fn page_dims(&self, z: u32) -> Option<(u32, u32)> {
        self.pages
            .iter()
            .find(|p| p.z == z)
            .map(|p| (p.width_px, p.height_px))
    }

    pub fn iter_cells(&self) -> impl Iterator<Item = &CellRecord> {
        self.cells.iter()
    }

    pub fn decode_to_text(&self) -> String {
        let ordered = self.ordered_cells();
        self.decode_cells_to_text(&ordered)
    }

    pub fn decode_page_to_text(&self, z: u32) -> String {
        let mut page_cells: Vec<_> = self.cells.iter().filter(|c| c.z == z).cloned().collect();
        page_cells.sort_by_key(|c| (c.y, c.x));
        self.decode_cells_to_text(&page_cells)
    }

    pub fn decode_cells_to_text(&self, cells: &[CellRecord]) -> String {
        let mut lines = Vec::with_capacity(cells.len());
        for cell in cells {
            if let Some(payload) = self.payload_for(&cell.code_id) {
                lines.push(payload.to_string());
            }
        }
        lines.join("\n")
    }

    pub fn cells_in_bbox(&self, z: u32, x0: i32, y0: i32, x1: i32, y1: i32) -> Vec<CellRecord> {
        let (min_x, max_x) = if x0 <= x1 { (x0, x1) } else { (x1, x0) };
        let (min_y, max_y) = if y0 <= y1 { (y0, y1) } else { (y1, y0) };
        let mut matches: Vec<_> = self
            .cells
            .iter()
            .filter(|cell| {
                if cell.z != z {
                    return false;
                }
                let cell_x1 = cell.x + cell.w as i32;
                let cell_y1 = cell.y + cell.h as i32;
                cell.x <= max_x && cell_x1 >= min_x && cell.y <= max_y && cell_y1 >= min_y
            })
            .cloned()
            .collect();
        matches.sort_by_key(|c| (c.y, c.x));
        matches
    }
}

pub fn hash_payload(payload: &str) -> CodeHash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(payload.as_bytes());
    let hash = hasher.finalize();
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(hash.as_bytes());
    bytes
}

impl Document {
    pub fn retain_dict_for_cells(&mut self) {
        let used: HashSet<_> = self.cells.iter().map(|c| c.code_id).collect();
        self.dict.retain(|code, _| used.contains(code));
    }

    pub fn numguard_mismatches(&self) -> Vec<NumGuardAlert> {
        self.numguard_mismatches_with_units(None)
    }

    pub fn numguard_mismatches_with_units(
        &self,
        whitelist: Option<&HashSet<String>>,
    ) -> Vec<NumGuardAlert> {
        let whitelist =
            whitelist.map(|set| set.iter().map(|s| s.to_lowercase()).collect::<HashSet<_>>());
        let mut alerts = Vec::new();
        for guard in &self.numguards {
            if let Some(ref allowed) = whitelist {
                if !guard.units.is_empty() && !allowed.contains(&guard.units.to_lowercase()) {
                    alerts.push(NumGuardAlert {
                        guard: guard.clone(),
                        observed: None,
                        issue: NumGuardIssue::UnitNotAllowed,
                    });
                    continue;
                }
            }
            let cell = self.cells.iter().find(|c| {
                c.z == guard.z && c.x.max(0) as u32 == guard.x && c.y.max(0) as u32 == guard.y
            });
            if let Some(cell) = cell {
                if let Some(payload) = self.payload_for(&cell.code_id) {
                    if let Some(actual) = crate::numguard::hash_digits_from_payload(payload) {
                        if actual != guard.sha1 {
                            alerts.push(NumGuardAlert {
                                guard: guard.clone(),
                                observed: Some(actual),
                                issue: NumGuardIssue::HashMismatch,
                            });
                        }
                        continue;
                    }
                    alerts.push(NumGuardAlert {
                        guard: guard.clone(),
                        observed: None,
                        issue: NumGuardIssue::MissingPayload,
                    });
                    continue;
                }
            }
            alerts.push(NumGuardAlert {
                guard: guard.clone(),
                observed: None,
                issue: NumGuardIssue::MissingCell,
            });
        }
        alerts
    }
}

#[derive(Debug, Clone)]
pub struct NumGuardAlert {
    pub guard: NumGuard,
    pub observed: Option<[u8; 20]>,
    pub issue: NumGuardIssue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumGuardIssue {
    MissingCell,
    MissingPayload,
    HashMismatch,
    UnitNotAllowed,
}

mod dict_serde {
    use super::CodeHash;
    use indexmap::IndexMap;
    use serde::ser::Serialize;
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(map: &IndexMap<CodeHash, String>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let as_vec: Vec<_> = map
            .iter()
            .map(|(code, payload)| (hex::encode(code), payload))
            .collect();
        as_vec.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<IndexMap<CodeHash, String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw: Vec<(String, String)> = Vec::deserialize(deserializer)?;
        let mut map = IndexMap::new();
        for (hex_code, payload) in raw {
            let bytes = hex::decode(&hex_code).map_err(D::Error::custom)?;
            if bytes.len() != 32 {
                return Err(D::Error::custom("invalid code hash length"));
            }
            let mut code = [0u8; 32];
            code.copy_from_slice(&bytes);
            map.insert(code, payload);
        }
        Ok(map)
    }
}

mod codehash_serde {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(hash: &[u8; 32], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(hash))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 32], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(D::Error::custom)?;
        if bytes.len() != 32 {
            return Err(D::Error::custom("invalid hash length"));
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes);
        Ok(hash)
    }
}

mod numhash_serde {
    use serde::{de::Error, Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(hash: &[u8; 20], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(hash))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 20], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(D::Error::custom)?;
        if bytes.len() != 20 {
            return Err(D::Error::custom("invalid sha1 length"));
        }
        let mut hash = [0u8; 20];
        hash.copy_from_slice(&bytes);
        Ok(hash)
    }
}
