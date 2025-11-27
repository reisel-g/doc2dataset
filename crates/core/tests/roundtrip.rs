use three_dcf_core::{hash_payload, CellRecord, CellType, Document, Header, PageInfo};

#[test]
fn document_roundtrip_bin() {
    let doc = sample_document();
    let bytes = doc.to_bytes().expect("serialize");
    let decoded = Document::from_bytes(&bytes).expect("decode");
    assert_eq!(doc.header.version, decoded.header.version);
    assert_eq!(doc.pages, decoded.pages);
    assert_eq!(doc.ordered_cells(), decoded.ordered_cells());

    for (code, payload) in &doc.dict {
        assert_eq!(payload, decoded.payload_for(code).unwrap());
    }
}

fn sample_document() -> Document {
    let mut doc = Document::new(Header::default());
    doc.add_page(PageInfo {
        z: 0,
        width_px: 800,
        height_px: 1000,
    });
    doc.add_page(PageInfo {
        z: 1,
        width_px: 800,
        height_px: 1000,
    });

    let payloads = ["Revenue", "Cost", "Net Income"];
    for (idx, payload) in payloads.iter().enumerate() {
        let hash = hash_payload(payload);
        doc.dict.insert(hash, payload.to_string());
        doc.cells.push(CellRecord {
            z: idx as u32,
            x: 50,
            y: 50 + (idx as i32 * 40),
            w: 700,
            h: 24,
            code_id: hash,
            rle: 0,
            cell_type: CellType::Text,
            importance: 100,
        });
    }
    doc
}

#[test]
fn bbox_decode_matches_subset() {
    let doc = sample_document();
    let cells = doc.cells_in_bbox(0, 0, 0, 800, 200);
    assert!(!cells.is_empty());
    let text_block = doc.decode_cells_to_text(&cells);
    assert!(text_block.contains("Revenue"));
    assert!(!text_block.contains("Net Income"));
}
