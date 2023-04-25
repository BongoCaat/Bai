use super::prelude::*;
use crate::{
    query::parser,
    semantic::Semantic,
    webserver::answer::{deduplicate_snippets, Snippet},
};
use tracing::error;

use qdrant_client::qdrant::{vectors, ScoredPoint};

#[derive(Deserialize)]
pub(super) struct Args {
    limit: u64,
    query: String,
}

#[derive(Serialize)]
pub(super) struct SemanticResponse {
    snippets: Vec<Snippet>,
}

impl super::ApiResponse for SemanticResponse {}

/// Get details of an indexed repository based on their id
//
#[utoipa::path(get, path = "/repos/indexed/:ref",
    responses(
        (status = 200, description = "Execute query successfully", body = SemanticResponse),
        (status = 400, description = "Bad request", body = EndpointError),
        (status = 500, description = "Server error", body = EndpointError),
    ),
)]
pub(super) async fn raw_chunks(
    Query(args): Query<Args>,
    Extension(semantic): Extension<Option<Semantic>>,
) -> impl IntoResponse {
    if let Some(semantic) = semantic {
        let Args { ref query, limit } = args;
        let parsed_query = parser::parse_nl(query).unwrap();
        let all_snippets: Vec<Snippet> = semantic
            .search(&parsed_query, 4 * limit) // heuristic
            .await
            .map_err(Error::internal)?
            .into_iter()
            .map(|r| {
                use qdrant_client::qdrant::{value::Kind, Value};

                // TODO: Can we merge with webserver/semantic.rs:L63?
                fn value_to_string(value: Value) -> String {
                    match value.kind.unwrap() {
                        Kind::StringValue(s) => s,
                        _ => panic!("got non-string value"),
                    }
                }

                fn extract_vector(point: &ScoredPoint) -> Vec<f32> {
                    if let Some(vectors) = &point.vectors {
                        if let Some(vectors::VectorsOptions::Vector(v)) = &vectors.vectors_options {
                            return v.data.clone();
                        }
                    }
                    panic!("got non-vector value");
                }

                let embedding = extract_vector(&r);

                let mut s = r.payload;

                Snippet {
                    lang: value_to_string(s.remove("lang").unwrap()),
                    repo_name: value_to_string(s.remove("repo_name").unwrap()),
                    repo_ref: value_to_string(s.remove("repo_ref").unwrap()),
                    relative_path: value_to_string(s.remove("relative_path").unwrap()),
                    text: value_to_string(s.remove("snippet").unwrap()),

                    start_line: value_to_string(s.remove("start_line").unwrap())
                        .parse::<usize>()
                        .unwrap(),
                    end_line: value_to_string(s.remove("end_line").unwrap())
                        .parse::<usize>()
                        .unwrap(),
                    start_byte: value_to_string(s.remove("start_byte").unwrap())
                        .parse::<usize>()
                        .unwrap(),
                    end_byte: value_to_string(s.remove("end_byte").unwrap())
                        .parse::<usize>()
                        .unwrap(),
                    score: r.score,
                    embedding,
                }
            })
            .collect();

        let query_target = parsed_query
            .target()
            .ok_or_else(|| Error::user("empty search"))?
            .to_string();
        let query_embedding = semantic.embed(&query_target).map_err(|e| {
            error!("failed to embed query: {}", e);
            Error::internal(e)
        })?;

        let snippets = deduplicate_snippets(all_snippets, query_embedding, limit as usize);
        Ok(json(SemanticResponse { snippets }))
    } else {
        Err(Error::new(
            ErrorKind::Configuration,
            "Qdrant not configured",
        ))
    }
}
