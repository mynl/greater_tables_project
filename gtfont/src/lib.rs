use fontdue::Font;
use pyo3::prelude::*;

#[pyclass]
struct FontMeasurer {
    font: Font,
}

#[pymethods]
impl FontMeasurer {
    #[new]
    fn new(font_bytes: &[u8]) -> PyResult<Self> {
        let font = Font::from_bytes(font_bytes, fontdue::FontSettings::default())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Font error: {e}")))?;
        Ok(Self { font })
    }

    fn measure(&self, text: &str, px: f32) -> f32 {
        text.chars()
            .map(|c| self.font.metrics(c, px).advance_width)
            .sum()
    }

    fn max_word_width(&self, text: &str, px: f32) -> f32 {
        text.split(' ')
            .map(|word| word.chars()
                .map(|c| self.font.metrics(c, px).advance_width)
                .sum::<f32>())
            .fold(0.0, f32::max)
    }
    fn measure_and_max_word(&self, text: &str, px: f32) -> (f32, f32) {
        let mut total: f32 = 0.0;
        let mut max_word: f32 = 0.0;

        for word in text.split(' ') {
            let word_width: f32 = word.chars()
                .map(|c| self.font.metrics(c, px).advance_width)
                .sum();
            total += word_width + self.font.metrics(' ', px).advance_width;
            max_word = max_word.max(word_width);
        }

        if text.ends_with(' ') {
            // trailing space is valid
        } else if total > 0.0 {
            total -= self.font.metrics(' ', px).advance_width;
        }

        (total, max_word)
    }


}

#[pymodule]
fn gtfont(_py: &Bound<'_, PyModule>) -> PyResult<()> {
    _py.add_class::<FontMeasurer>()?;
    Ok(())
}
