use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

pub fn convert_rtf(path: &Path) -> Result<String> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read RTF file {}", path.display()))?;
    let mut out = String::new();
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        out.push_str(&format!("# {} (rtf)\n\n", stem));
    }
    let rendered = rtf_to_text(&raw);
    out.push_str(rendered.trim());
    out.push('\n');
    Ok(out)
}

fn rtf_to_text(input: &str) -> String {
    let mut out = String::new();
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '{' | '}' => continue,
            '\\' => {
                if let Some(next) = chars.peek().copied() {
                    match next {
                        '\\' | '{' | '}' => {
                            out.push(next);
                            chars.next();
                        }
                        '\'' => {
                            chars.next();
                            let hi = chars.next();
                            let lo = chars.next();
                            if let (Some(h), Some(l)) = (hi, lo) {
                                if let Ok(value) = u8::from_str_radix(&format!("{}{}", h, l), 16) {
                                    out.push(value as char);
                                }
                            }
                        }
                        _ => {
                            consume_control_word(&mut chars);
                        }
                    }
                }
            }
            '\r' => {
                out.push('\n');
            }
            '\n' => {
                out.push('\n');
            }
            other => out.push(other),
        }
    }
    out
}

fn consume_control_word<I>(chars: &mut std::iter::Peekable<I>)
where
    I: Iterator<Item = char>,
{
    while let Some(&c) = chars.peek() {
        if c.is_ascii_alphabetic() {
            chars.next();
        } else {
            break;
        }
    }
    if let Some(&c) = chars.peek() {
        if c == '-' || c.is_ascii_digit() {
            chars.next();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    chars.next();
                } else {
                    break;
                }
            }
        }
    }
    if let Some(&c) = chars.peek() {
        if c == ' ' {
            chars.next();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn rtf_is_converted_to_plain_text() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"{\\rtf1\\ansi This is \\b bold\\b0 text}")
            .unwrap();
        let markdown = convert_rtf(file.path()).unwrap();
        assert!(markdown.contains("This is"));
        assert!(markdown.contains("bold"));
    }
}
