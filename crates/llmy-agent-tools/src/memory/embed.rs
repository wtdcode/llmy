use std::{
    fmt::Display,
    ops::Deref,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use color_eyre::eyre::eyre;
use fastembed::{EmbeddingModel, ModelTrait, TextEmbedding, TextInitOptions};
use llmy_types::error::LLMYError;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embeding(Vec<f64>);

impl Embeding {
    pub fn calculate_similarity(&self, targets: &[Embeding]) -> Vec<f64> {
        targets
            .iter()
            .map(|passage| self.cosine_similarity(passage).unwrap_or(0.0))
            .collect()
    }

    pub fn cosine_similarity(&self, right: &Self) -> Option<f64> {
        if self.0.len() != right.0.len() || self.0.is_empty() {
            return None;
        }

        let dot = self
            .0
            .iter()
            .zip(right.0.iter())
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let left_norm = self.0.iter().map(|value| value * value).sum::<f64>().sqrt();
        let right_norm = right
            .0
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();

        if left_norm == 0.0 || right_norm == 0.0 {
            return Some(0.0);
        }

        Some((dot / (left_norm * right_norm)).clamp(0.0, 1.0))
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimilarityModelConfig {
    pub model: EmbeddingModel,
    pub cache_dir: Option<PathBuf>,
}

impl Display for SimilarityModelConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "SimilarityConfig(model={},cache_dir={})",
            EmbeddingModel::get_model_info(&self.model)
                .unwrap()
                .model_code,
            self.cache_dir
                .as_ref()
                .map(|v| v.display().to_string())
                .unwrap_or_else(|| "None".to_string())
        ))
    }
}

pub struct SimilarityModelInner {
    embedding: Mutex<TextEmbedding>,
    pub config: SimilarityModelConfig,
}

impl SimilarityModelInner {
    pub fn batch_embed(&self, queries: Vec<String>) -> Result<Vec<Embeding>, LLMYError> {
        let mut model = self
            .embedding
            .lock()
            .map_err(|_| eyre!("local embedding model lock was poisoned"))?;
        let embeds = model
            .embed(queries, None)
            .map_err(|error| eyre!("failed to compute local title embeddings: {error}"))?;

        Ok(embeds
            .into_iter()
            .map(|v| v.into_iter().map(f64::from).collect::<Vec<f64>>())
            .map(|v| Embeding(v))
            .collect())
    }
}

#[derive(Clone)]
pub struct SimilarityModel {
    pub inner: Arc<SimilarityModelInner>,
}

impl Deref for SimilarityModel {
    type Target = SimilarityModelInner;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl SimilarityModel {
    pub async fn new(config: SimilarityModelConfig) -> Result<Self, LLMYError> {
        tracing::info!("Initializing local title embedding model {}", &config);
        if let Some(cache_dir) = &config.cache_dir {
            tracing::info!("Using title embedding cache dir {}", cache_dir.display());
        }

        let config_for_init = config.clone();
        let embedding = tokio::task::spawn_blocking(move || {
            let mut options = TextInitOptions::new(config_for_init.model.clone());
            if let Some(cache_dir) = config_for_init.cache_dir.clone() {
                options = options.with_cache_dir(cache_dir);
            }
            TextEmbedding::try_new(options).map_err(|error| {
                eyre!(
                    "failed to initialize local embedding model {}: {}",
                    config_for_init,
                    error
                )
            })
        })
        .await
        .map_err(|error| eyre!("local embedding initialization panicked: {error}"))??;

        Ok(Self {
            inner: Arc::new(SimilarityModelInner {
                embedding: Mutex::new(embedding),
                config,
            }),
        })
    }

    pub async fn batch_embed(&self, queries: Vec<String>) -> Result<Vec<Embeding>, LLMYError> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || inner.batch_embed(queries))
            .await
            .expect("batch embed paniced")
    }

    pub async fn batch_similarity(
        &self,
        query: String,
        passages: Vec<String>,
    ) -> Result<Vec<f64>, LLMYError> {
        if query.is_empty() {
            return Ok(vec![0.0; passages.len()]);
        }

        if passages.is_empty() {
            return Ok(Vec::new());
        }

        let exact_scores = passages
            .iter()
            .map(|passage| {
                if passage.is_empty() {
                    Some(0.0)
                } else if *passage == query {
                    Some(1.0)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if exact_scores.iter().all(Option::is_some) {
            return Ok(exact_scores
                .into_iter()
                .map(|score| score.unwrap_or(0.0))
                .collect());
        }

        let query_input = format!("query: {}", &query);
        let pending_indices = passages
            .iter()
            .enumerate()
            .filter_map(|(index, passage)| {
                exact_scores[index]
                    .is_none()
                    .then(|| (index, format!("passage: {}", &passage)))
            })
            .collect::<Vec<_>>();
        let pending_inputs = pending_indices
            .iter()
            .map(|(_, input)| input.clone())
            .collect::<Vec<_>>();

        let mut inputs = Vec::with_capacity(pending_inputs.len() + 1);
        inputs.push(query_input);
        inputs.extend(pending_inputs);

        let embeddings = self.batch_embed(inputs).await?;

        let Some((query_embedding, candidate_embeddings)) = embeddings.split_first() else {
            return Err(eyre!("local embedding model returned no query embedding").into());
        };

        let similarities = query_embedding.calculate_similarity(candidate_embeddings);

        let mut scores = vec![0.0; passages.len()];
        for (exact_score, slot) in exact_scores.iter().zip(scores.iter_mut()) {
            if let Some(exact_score) = exact_score {
                *slot = *exact_score;
            }
        }

        for ((index, _), similarity) in pending_indices.into_iter().zip(similarities.into_iter()) {
            scores[index] = similarity;
        }

        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;
    use tempfile::TempDir;
    use tokio::sync::OnceCell;

    static CACHE_DIR: LazyLock<TempDir> =
        LazyLock::new(|| tempfile::tempdir().expect("failed to create temp cache dir"));
    static MODEL: OnceCell<SimilarityModel> = OnceCell::const_new();

    async fn shared_model() -> SimilarityModel {
        MODEL
            .get_or_init(|| async {
                SimilarityModel::new(SimilarityModelConfig {
                    cache_dir: Some(CACHE_DIR.path().to_path_buf()),
                    ..Default::default()
                })
                .await
                .expect("failed to initialize shared similarity model")
            })
            .await
            .clone()
    }

    #[tokio::test(flavor = "current_thread")]
    async fn batch_embed_uses_shared_temp_cache_dir() {
        let model = shared_model().await;

        assert_eq!(model.config.cache_dir.as_deref(), Some(CACHE_DIR.path()));

        let embeddings = model
            .batch_embed(vec![
                "query: rust async runtime".to_string(),
                "passage: tokio futures executor".to_string(),
            ])
            .await
            .expect("failed to compute embeddings");

        assert_eq!(embeddings.len(), 2);
        assert!(!embeddings[0].0.is_empty());
        assert_eq!(embeddings[0].0.len(), embeddings[1].0.len());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn batch_similarity_short_circuits_exact_and_empty_passages() {
        let model = shared_model().await;
        let query = "rust ownership".to_string();

        let scores = model
            .batch_similarity(query.clone(), vec![query, String::new()])
            .await
            .expect("failed to compute similarity scores");

        assert_eq!(scores, vec![1.0, 0.0]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn batch_similarity_matches_batch_embed_and_calculate_similarity() {
        let model = shared_model().await;
        let query = "rust async runtime".to_string();
        let passages = vec![
            "tokio runtime for async rust futures".to_string(),
            "banana bread recipe with walnuts".to_string(),
        ];

        let scores = model
            .batch_similarity(query.clone(), passages.clone())
            .await
            .expect("failed to compute similarity scores");

        let mut inputs = Vec::with_capacity(passages.len() + 1);
        inputs.push(format!("query: {query}"));
        inputs.extend(passages.iter().map(|passage| format!("passage: {passage}")));

        let cloned_model = model.clone();
        let embeddings = cloned_model
            .batch_embed(inputs)
            .await
            .expect("failed to compute embeddings for expected similarities");
        let (query_embedding, candidate_embeddings) = embeddings
            .split_first()
            .expect("expected query embedding and candidate embeddings");
        let expected_scores = query_embedding.calculate_similarity(candidate_embeddings);

        assert_eq!(scores, expected_scores);
        assert!(
            scores[0] > scores[1],
            "expected related passage to rank higher"
        );
    }
}
