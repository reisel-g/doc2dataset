use proptest::prelude::*;
use three_dcf_core::{hash_payload, CellRecord, CellType, Document, Header, PageInfo};

proptest! {
    #[test]
    fn deltas_roundtrip(cells in cell_vec()) {
        let mut doc = Document::new(Header::default());
        let mut seen_pages = std::collections::BTreeSet::new();
        for spec in &cells {
            if seen_pages.insert(spec.z) {
                doc.add_page(PageInfo { z: spec.z, width_px: 800, height_px: 1000 });
            }
            let hash = hash_payload(&spec.payload);
            doc.dict.insert(hash, spec.payload.clone());
            doc.cells.push(CellRecord {
                z: spec.z,
                x: spec.x,
                y: spec.y,
                w: spec.w,
                h: spec.h,
                code_id: hash,
                rle: 0,
                cell_type: spec.cell_type,
                importance: spec.importance,
            });
        }

        let bytes = doc.to_bytes().expect("serialize");
        let decoded = Document::from_bytes(&bytes).expect("decode");
        prop_assert_eq!(doc.ordered_cells(), decoded.ordered_cells());
        for (code, payload) in &doc.dict {
            prop_assert_eq!(payload, decoded.payload_for(code).unwrap());
        }
    }
}

#[derive(Clone, Debug)]
struct CellSpec {
    z: u32,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    cell_type: CellType,
    importance: u8,
    payload: String,
}

fn cell_vec() -> impl Strategy<Value = Vec<CellSpec>> {
    prop::collection::vec(cell_spec(), 1..20)
}

fn cell_spec() -> impl Strategy<Value = CellSpec> {
    (
        0u32..5,
        -500i32..500,
        -500i32..500,
        10u32..600,
        10u32..200,
        prop_oneof![
            Just(CellType::Text),
            Just(CellType::Table),
            Just(CellType::Header),
            Just(CellType::Footer),
        ],
        any::<u8>(),
        "[A-Za-z0-9 .,%-]{3,64}".prop_map(|s| s.to_string()),
    )
        .prop_map(|(z, x, y, w, h, cell_type, importance, payload)| CellSpec {
            z,
            x,
            y,
            w,
            h,
            cell_type,
            importance,
            payload,
        })
}
