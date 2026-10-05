use super::{Article, is_web_link, save_json};
use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, io::Read, path::Path};

const FEED_URL: &str = "https://www.producthunt.com/feed";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Product {
    pub id: String,
    pub article: Article,
}

/// Products can be featured days after submission: cache the daily feed snapshot.
#[derive(Debug, Serialize, Deserialize)]
pub struct DailyCache {
    pub day: NaiveDate,
    pub fetched_at: DateTime<Utc>,
    pub products: Vec<Product>,
    #[serde(default)]
    pub commented: HashSet<String>,
}

impl DailyCache {
    pub fn next_product(&self) -> Option<&Product> {
        self.products
            .iter()
            .find(|p| !self.commented.contains(&p.id))
    }

    pub fn remaining(&self) -> usize {
        self.products
            .iter()
            .filter(|p| !self.commented.contains(&p.id))
            .count()
    }
}

#[derive(Deserialize)]
struct Feed {
    #[serde(default, rename = "entry")]
    entries: Vec<AtomEntry>,
}

#[derive(Deserialize)]
struct AtomEntry {
    id: String,
    title: String,
    #[serde(default, rename = "link")]
    links: Vec<AtomLink>,
    #[serde(default)]
    published: String,
    #[serde(default)]
    content: AtomContent,
    #[serde(default)]
    summary: AtomContent,
}

#[derive(Default, Deserialize)]
struct AtomContent {
    #[serde(default, rename = "$text")]
    text: String,
}

#[derive(Deserialize)]
struct AtomLink {
    #[serde(default, rename = "@rel")]
    rel: String,
    #[serde(rename = "@href")]
    href: String,
}

fn description(html: &str) -> String {
    // The second paragraph contains Discussion/Link navigation, not a description.
    let paragraph = regex::Regex::new(r"(?is)<p\b[^>]*>(.*?)</p>").unwrap();
    let text = paragraph
        .captures(html)
        .map(|c| c[1].to_owned())
        .unwrap_or_else(|| html.to_owned());
    let tags = regex::Regex::new(r"(?s)<[^>]*>").unwrap();
    let text = tags.replace_all(&text, " ");
    let text = text
        .replace("&nbsp;", " ")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–");
    let decoded = quick_xml::escape::unescape(&text).unwrap_or_else(|_| text.clone().into());
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn parse_products(xml: &str) -> Result<Vec<Product>> {
    let mut reader = quick_xml::Reader::from_str(xml);
    loop {
        match reader.read_event().context("Invalid Product Hunt XML")? {
            quick_xml::events::Event::Start(root) | quick_xml::events::Event::Empty(root) => {
                anyhow::ensure!(
                    root.local_name().as_ref() == b"feed",
                    "Expected a Product Hunt Atom feed"
                );
                break;
            }
            quick_xml::events::Event::Eof => anyhow::bail!("Empty Product Hunt feed"),
            _ => {}
        }
    }
    let feed: Feed = quick_xml::de::from_str(xml).context("Invalid Product Hunt Atom feed")?;
    let mut ids = HashSet::new();
    Ok(feed
        .entries
        .into_iter()
        .filter_map(|entry| {
            let link = entry
                .links
                .into_iter()
                .find(|l| (l.rel == "alternate" || l.rel.is_empty()) && is_web_link(&l.href))?;
            let title = entry.title.split_whitespace().collect::<Vec<_>>().join(" ");
            let description = description(if entry.content.text.is_empty() {
                &entry.summary.text
            } else {
                &entry.content.text
            });
            if title.is_empty()
                || description.is_empty()
                || entry.id.trim().is_empty()
                || !ids.insert(entry.id.clone())
            {
                return None;
            }
            Some(Product {
                id: entry.id,
                article: Article {
                    title,
                    link: link.href,
                    published: DateTime::parse_from_rfc3339(&entry.published)
                        .ok()
                        .map(|d| d.with_timezone(&Utc)),
                    description,
                },
            })
        })
        .collect())
}

pub fn fetch(agent: &ureq::Agent, day: NaiveDate) -> Result<DailyCache> {
    const MAX_BYTES: u64 = 4 * 1024 * 1024;
    let response = agent
        .get(FEED_URL)
        .set("User-Agent", "YumonRSS/0.1")
        .set("Accept", "application/atom+xml")
        .call()
        .context("Could not fetch Product Hunt")?;
    let mut bytes = Vec::new();
    response
        .into_reader()
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)?;
    anyhow::ensure!(
        bytes.len() as u64 <= MAX_BYTES,
        "Product Hunt feed exceeds 4 MB"
    );
    let products = parse_products(std::str::from_utf8(&bytes).context("Feed is not UTF-8")?)?;
    anyhow::ensure!(
        !products.is_empty(),
        "Product Hunt has no products with descriptions yet"
    );
    Ok(DailyCache {
        day,
        fetched_at: Utc::now(),
        products,
        commented: HashSet::new(),
    })
}

pub fn load(path: &Path, day: NaiveDate) -> Result<Option<DailyCache>> {
    match std::fs::read(path) {
        Ok(bytes) => {
            let cache: DailyCache =
                serde_json::from_slice(&bytes).context("Invalid Product Hunt cache")?;
            anyhow::ensure!(
                cache.products.iter().all(|p| is_web_link(&p.article.link)),
                "Invalid cached product URL"
            );
            Ok((cache.day == day).then_some(cache))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn save(path: &Path, cache: &DailyCache) -> Result<()> {
    save_json(path, cache)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FEED: &str = r#"<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>post/1</id><title>A &amp; B</title><published>2026-09-23T12:54:18-07:00</published>
      <link rel="self" href="https://example.com/api"/><link rel="alternate" href="https://example.com/product"/>
      <content type="html">&lt;p&gt; Build &amp;amp; sell &lt;b&gt;ideas&lt;/b&gt; &amp;#128640; &lt;/p&gt;&lt;p&gt;Discussion | Link&lt;/p&gt;</content></entry>
      <entry><id>post/1</id><title>Duplicate</title><link href="https://example.com/product"/><summary>Duplicate</summary></entry>
      <entry><id>post/2</id><title>Unsafe</title><link href="javascript:alert(1)"/><summary>Bad</summary></entry>
      <entry><id>post/3</id><title>No description</title><link href="https://example.com/empty"/></entry>
      <entry><id>post/4</id><title>Second launch</title><link href="https://example.com/product"/><summary>New idea</summary></entry>
    </feed>"#;

    #[test]
    fn parses_daily_snapshot_descriptions_and_distinct_launches() {
        let products = parse_products(FEED).unwrap();
        assert_eq!(products.len(), 2);
        assert_eq!(products[0].article.description, "Build & sell ideas 🚀");
        assert_eq!(products[0].article.title, "A & B");
        assert_eq!(products[0].article.link, "https://example.com/product");
        assert!(products[0].article.published.is_some());
        assert!(parse_products("<html>Oops</html>").is_err());
    }

    #[test]
    fn cache_resumes_queue_and_expires_on_the_next_day() {
        let day = NaiveDate::from_ymd_opt(2026, 10, 5).unwrap();
        let path =
            std::env::temp_dir().join(format!("yumon-products-{}.json", uuid::Uuid::new_v4()));
        assert!(load(&path, day).unwrap().is_none());
        let mut cache = DailyCache {
            day,
            fetched_at: Utc::now(),
            products: parse_products(FEED).unwrap(),
            commented: HashSet::new(),
        };
        cache
            .commented
            .insert(cache.next_product().unwrap().id.clone());
        save(&path, &cache).unwrap();
        let restored = load(&path, day).unwrap().unwrap();
        assert_eq!(restored.next_product().unwrap().id, "post/4");
        assert_eq!(restored.remaining(), 1);
        assert!(load(&path, day.succ_opt().unwrap()).unwrap().is_none());
        cache.commented.insert("post/4".into());
        save(&path, &cache).unwrap();
        assert!(load(&path, day).unwrap().unwrap().next_product().is_none());
        std::fs::write(&path, "broken").unwrap();
        assert!(load(&path, day).is_err());
        std::fs::remove_file(path).unwrap();
    }
}
