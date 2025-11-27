use crate::document::Document;
use crate::error::Result;

#[derive(Debug, Default)]
pub struct Decoder;

impl Decoder {
    pub fn new() -> Self {
        Self
    }

    pub fn to_text(&self, document: &Document) -> Result<String> {
        Ok(document.decode_to_text())
    }

    pub fn page_to_text(&self, document: &Document, z: u32) -> Result<String> {
        Ok(document.decode_page_to_text(z))
    }

    pub fn bbox_to_text(
        &self,
        document: &Document,
        z: u32,
        x0: i32,
        y0: i32,
        x1: i32,
        y1: i32,
    ) -> Result<String> {
        let cells = document.cells_in_bbox(z, x0, y0, x1, y1);
        Ok(document.decode_cells_to_text(&cells))
    }
}
