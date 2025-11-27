fn main() {
    let mut features = vec!["base".to_string()];
    if cfg!(feature = "pdfium") {
        features.push("pdfium".to_string());
    }
    if cfg!(feature = "ocr") {
        features.push("ocr".to_string());
    }
    println!("cargo:rustc-env=THREE_DCF_FEATURES={}", features.join(", "));
}
