use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{io::Read, path::Path, time::Duration};

pub const POLL_INTERVAL: Duration = Duration::from_secs(15 * 60);
pub const HISTORY_LIMIT: usize = 200;

pub struct Source {
    pub name: &'static str,
    pub url: &'static str,
}

// https://news.ycombinator.com/rss
// https://www.theverge.com/rss/index.xml
// https://www.wired.com/feed/rss
// https://www.politico.com/rss

// Edit this list to change Yumon's news sources. These are RSS 2.0 feeds.
pub const SOURCES: &[Source] = &[
    // Source {
    //     name: "BBC World",
    //     url: "https://feeds.bbci.co.uk/news/world/rss.xml",
    // },
    // Source {
    //     name: "Guardian Science",
    //     url: "https://www.theguardian.com/science/rss",
    // },
    // Source {
    //     name: "NASA",
    //     url: "https://www.nasa.gov/feed/",
    // },
    Source {
        name: "Hacker News",
        url: "https://news.ycombinator.com/rss",
    },
    // Source {
    //     name: "The Verge",
    //     url: "https://www.theverge.com/rss/index.xml",
    // },
    Source {
        name: "Wired",
        url: "https://www.wired.com/feed/rss",
    },
    Source {
        name: "Politico",
        url: "https://rss.politico.com/politics-news.xml",
    },
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Article {
    pub title: String,
    pub link: String,
    pub published: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub source: String,
    pub article: Article,
    pub comment: String,
    pub received: DateTime<Utc>,
}

#[derive(Deserialize)]
struct Feed {
    channel: Channel,
}

#[derive(Deserialize)]
struct Channel {
    #[serde(default, rename = "item")]
    items: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    #[serde(default)]
    title: String,
    #[serde(default)]
    link: String,
    #[serde(default, rename = "pubDate")]
    published: String,
}

pub fn is_web_link(link: &str) -> bool {
    let Ok(url) = link.parse::<wry::http::Uri>() else {
        return false;
    };
    matches!(url.scheme_str(), Some("http" | "https")) && url.host().is_some()
}

/// Dates determine the newest item; ties/missing dates retain the feed's order.
pub fn latest_article(xml: &str) -> Result<Option<Article>> {
    let feed: Feed = quick_xml::de::from_str(xml).context("Invalid RSS feed")?;
    let mut articles: Vec<Article> = feed
        .channel
        .items
        .into_iter()
        .filter_map(|item| {
            let title = item.title.split_whitespace().collect::<Vec<_>>().join(" ");
            let link = item.link.trim().to_owned();
            if title.is_empty() || !is_web_link(&link) {
                return None;
            }
            Some(Article {
                title,
                link,
                published: DateTime::parse_from_rfc2822(item.published.trim())
                    .ok()
                    .map(|d| d.with_timezone(&Utc)),
            })
        })
        .collect();
    articles.sort_by(|a, b| b.published.cmp(&a.published));
    Ok(articles.into_iter().next())
}

pub fn fetch_latest(agent: &ureq::Agent, source: &Source) -> Result<Option<Article>> {
    const MAX_BYTES: u64 = 4 * 1024 * 1024;
    let response = agent
        .get(source.url)
        .set("User-Agent", "YumonRSS/0.1")
        .set("Accept", "application/rss+xml, application/xml, text/xml")
        .call()
        .with_context(|| format!("Could not fetch {}", source.name))?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(bytes.len() as u64 <= MAX_BYTES, "RSS feed exceeds 4 MB");
    latest_article(std::str::from_utf8(&bytes).context("RSS feed is not UTF-8")?)
}

pub fn already_seen(history: &[Entry], article: &Article) -> bool {
    history
        .iter()
        .any(|entry| entry.article.link == article.link)
}

pub fn load_history(path: &Path) -> Result<Vec<Entry>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let mut entries: Vec<Entry> =
                serde_json::from_slice(&bytes).context("Invalid RSS history")?;
            entries.retain(|e| is_web_link(&e.article.link));
            if entries.len() > HISTORY_LIMIT {
                entries.drain(..entries.len() - HISTORY_LIMIT);
            }
            Ok(entries)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(e.into()),
    }
}

pub fn save_history(path: &Path, history: &[Entry]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut temporary = path.as_os_str().to_os_string();
    temporary.push(".tmp");
    let temporary = std::path::PathBuf::from(temporary);
    std::fs::write(&temporary, serde_json::to_vec_pretty(history)?)?;
    std::fs::rename(&temporary, path)
        .with_context(|| format!("Could not save history to {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_newest_and_decodes_entities_and_cdata() {
        let xml = r#"<rss version="2.0"><channel><title>Feed</title>
            <item><title>Old &amp; safe</title><link>https://example.com/old</link><pubDate>Mon, 05 Oct 2026 08:00:00 GMT</pubDate></item>
            <item><title><![CDATA[New <discovery> & news]]></title><link>https://example.com/new?a=1&amp;b=2</link><pubDate>Mon, 05 Oct 2026 09:00:00 GMT</pubDate></item>
        </channel></rss>"#;
        let article = latest_article(xml).unwrap().unwrap();
        assert_eq!(article.title, "New <discovery> & news");
        assert_eq!(article.link, "https://example.com/new?a=1&b=2");
        assert_eq!(article.published.unwrap().timestamp(), 1791190800);
    }

    #[test]
    fn missing_dates_preserve_order_and_invalid_items_are_skipped() {
        let xml = r#"<rss><channel>
            <item><title>Unsafe</title><link>javascript:alert(1)</link></item>
            <item><title>  First   valid </title><link>https://example.com/first</link></item>
            <item><title>Second</title><link>https://example.com/second</link><pubDate>bad date</pubDate></item>
        </channel></rss>"#;
        assert_eq!(latest_article(xml).unwrap().unwrap().title, "First valid");
        assert!(latest_article("<rss><channel/></rss>").unwrap().is_none());
        assert!(latest_article("<html>not RSS</html>").is_err());
        assert!(!is_web_link("https:///"));
        assert!(!is_web_link("file:///secret"));
    }

    #[test]
    fn duplicate_detection_uses_article_link() {
        let article = Article {
            title: "News".into(),
            link: "https://example.com/news".into(),
            published: None,
        };
        let history = vec![Entry {
            source: "Test".into(),
            article: article.clone(),
            comment: "Oh!".into(),
            received: Utc::now(),
        }];
        let mut updated = article;
        updated.title = "Updated headline".into();
        assert!(already_seen(&history, &updated));
        updated.link = "https://example.com/another".into();
        assert!(!already_seen(&history, &updated));
    }

    #[test]
    fn history_round_trip_replaces_existing_file_and_limits_entries() {
        let path =
            std::env::temp_dir().join(format!("yumon-rss-test-{}.json", uuid::Uuid::new_v4()));
        assert!(load_history(&path).unwrap().is_empty());
        let history: Vec<Entry> = (0..HISTORY_LIMIT + 2)
            .map(|n| Entry {
                source: "Test".into(),
                article: Article {
                    title: format!("News {n}"),
                    link: format!("https://example.com/{n}"),
                    published: None,
                },
                comment: "A thought".into(),
                received: Utc::now(),
            })
            .collect();
        save_history(&path, &history).unwrap();
        let restored = load_history(&path).unwrap();
        assert_eq!(restored.len(), HISTORY_LIMIT);
        assert_eq!(restored[0].article.title, "News 2");
        save_history(&path, &restored).unwrap();
        assert_eq!(load_history(&path).unwrap().len(), HISTORY_LIMIT);
        std::fs::write(&path, b"broken JSON").unwrap();
        assert!(load_history(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
