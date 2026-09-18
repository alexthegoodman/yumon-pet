//! Deterministic ruler simulation, independent of rendering and language inference.
pub const NAMES: [&str; 10] = [
    "Ember", "Ripple", "Fern", "Sol", "Maple", "Clover", "Pip", "Willow", "Peach", "Moss",
];
pub const COLORS: [[u8; 3]; 3] = [[75, 155, 235], [235, 165, 65], [160, 100, 215]];
pub const SEASON_SECONDS: f32 = 25.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjectKind {
    Workshop,
    Garden,
    Observatory,
}
impl ProjectKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Workshop => "Workshop",
            Self::Garden => "Community garden",
            Self::Observatory => "Observatory",
        }
    }
    pub fn cost(self) -> u64 {
        match self {
            Self::Workshop => 90,
            Self::Garden => 65,
            Self::Observatory => 120,
        }
    }
    pub fn benefit(self) -> &'static str {
        match self {
            Self::Workshop => "+5 earnings per citizen each season",
            Self::Garden => "+3 public satisfaction",
            Self::Observatory => "+2 research each season; inventions improve earnings",
        }
    }
}
#[derive(Clone, Debug)]
pub struct Kingdom {
    pub name: &'static str,
    pub treasury: u64,
    pub tax: u8,
    pub satisfaction: i32,
    pub research: u64,
    pub last_tax: u64,
}
#[derive(Clone, Debug)]
pub struct Citizen {
    pub kingdom: usize,
    pub savings: u64,
    pub memory: String,
}
#[derive(Clone, Debug)]
pub struct Parcel {
    pub owner: Option<usize>,
    pub kind: ProjectKind,
    pub reserved: bool,
    pub project: Option<usize>,
}
#[derive(Clone, Debug)]
pub struct Project {
    pub citizen: usize,
    pub kind: ProjectKind,
    pub parcel: Option<usize>,
    pub remaining: u8,
    pub completed: bool,
}
#[derive(Clone, Debug)]
pub struct LandOffer {
    pub seller: usize,
    pub parcel: usize,
    pub price: u64,
    pub expires: u64,
}
pub struct Realm {
    pub kingdoms: Vec<Kingdom>,
    pub citizens: Vec<Citizen>,
    pub parcels: Vec<Parcel>,
    pub projects: Vec<Project>,
    pub season: u64,
    pub elapsed: f32,
    pub events: Vec<String>,
    pub trade: [bool; 3],
    pub relations: [i32; 3],
    pub offer: Option<LandOffer>,
}
impl Realm {
    pub fn new(seed: u64) -> Self {
        let mut parcels: Vec<_> = (0..16)
            .map(|i| Parcel {
                owner: None,
                kind: match (i as u64 + seed % 3) % 3 {
                    0 => ProjectKind::Workshop,
                    1 => ProjectKind::Garden,
                    _ => ProjectKind::Observatory,
                },
                reserved: false,
                project: None,
            })
            .collect();
        for (civ, ids) in [[0, 1], [3, 7], [14, 15]].iter().enumerate() {
            for &id in ids {
                parcels[id].owner = Some(civ);
            }
        }
        let citizens = (0..10)
            .map(|i| Citizen {
                kingdom: Self::citizen_kingdom(i),
                savings: 25,
                memory: "Hoping the new ruler will back our ambitions.".into(),
            })
            .collect();
        let projects = (0..10)
            .map(|i| Project {
                citizen: i,
                kind: parcels[[0, 1, 0, 1, 3, 7, 3, 14, 15, 14][i]].kind,
                parcel: None,
                remaining: 0,
                completed: false,
            })
            .collect();
        Self {
            kingdoms: ["Brookhaven (you)", "Amber League", "Violet Assembly"]
                .iter()
                .map(|&name| Kingdom {
                    name,
                    treasury: 180,
                    tax: 15,
                    satisfaction: 65,
                    research: 0,
                    last_tax: 0,
                })
                .collect(),
            citizens,
            parcels,
            projects,
            season: 1,
            elapsed: 0.0,
            events: vec![
                "Your first audience is ready. Citizens are asking for project funding.".into(),
            ],
            trade: [false; 3],
            relations: [0, 20, 20],
            offer: None,
        }
    }
    pub fn citizen_kingdom(id: usize) -> usize {
        if id < 4 {
            0
        } else if id < 7 {
            1
        } else {
            2
        }
    }
    pub fn center(id: usize) -> (f32, f32) {
        (
            -22.5 + (id % 4) as f32 * 15.0,
            -22.5 + (id / 4) as f32 * 15.0,
        )
    }
    pub fn parcel_at(x: f32, z: f32) -> Option<usize> {
        if !(-30.0..30.0).contains(&x) || !(-30.0..30.0).contains(&z) {
            return None;
        }
        Some(((z + 30.0) / 15.0) as usize * 4 + ((x + 30.0) / 15.0) as usize)
    }
    fn event(&mut self, text: String) {
        self.events.insert(0, text);
        self.events.truncate(24);
    }
    pub fn adjacent(&self, civ: usize, id: usize) -> bool {
        id < 16
            && (0..16).any(|other| {
                self.parcels[other].owner == Some(civ)
                    && (other % 4).abs_diff(id % 4) + (other / 4).abs_diff(id / 4) == 1
            })
    }
    pub fn land_price(&self, id: usize) -> u64 {
        85 + self.parcels[id].kind.cost() / 3
    }
    pub fn buy_land(&mut self, civ: usize, id: usize) -> Result<(), &'static str> {
        if civ >= self.kingdoms.len() || id >= 16 {
            return Err("Unknown kingdom or parcel.");
        }
        if self.parcels[id].owner.is_some() {
            return Err("Owned land requires a diplomatic offer.");
        }
        if !self.adjacent(civ, id) {
            return Err("New territory must share a border with your kingdom.");
        }
        let price = self.land_price(id);
        if self.kingdoms[civ].treasury < price {
            return Err("Your treasury cannot afford this parcel yet.");
        }
        self.kingdoms[civ].treasury -= price;
        self.parcels[id].owner = Some(civ);
        self.event(format!(
            "{} purchased parcel {} for {} coins.",
            self.kingdoms[civ].name,
            id + 1,
            price
        ));
        Ok(())
    }
    pub fn site_for(&self, project: usize) -> Option<usize> {
        let p = self.projects.get(project)?;
        let civ = self.citizens[p.citizen].kingdom;
        self.parcels
            .iter()
            .position(|s| {
                s.owner == Some(civ) && s.kind == p.kind && !s.reserved && s.project.is_none()
            })
            .or_else(|| {
                self.parcels.iter().position(|s| {
                    s.owner == Some(civ)
                        && s.kind == p.kind
                        && !s.reserved
                        && s.project.is_some_and(|id| self.projects[id].completed)
                })
            })
    }
    pub fn fund(&mut self, civ: usize, project: usize) -> Result<(), &'static str> {
        let p = self.projects.get(project).ok_or("Unknown proposal.")?;
        if self.citizens[p.citizen].kingdom != civ {
            return Err("Only your own citizens may receive your grants.");
        }
        if p.parcel.is_some() {
            return Err("This project has already received funding.");
        }
        let site = self
            .site_for(project)
            .ok_or("Buy suitable land or release a reserved parcel first.")?;
        let cost = p.kind.cost();
        if self.kingdoms[civ].treasury < cost {
            return Err("Not enough treasury funds.");
        }
        self.kingdoms[civ].treasury -= cost;
        self.parcels[site].project = Some(project);
        let p = &mut self.projects[project];
        p.parcel = Some(site);
        p.remaining = 2;
        self.citizens[p.citizen].memory = format!(
            "My ruler funded my {} on parcel {}!",
            p.kind.name(),
            site + 1
        );
        let text = format!(
            "{} began a {} on parcel {}. Ready in two seasons.",
            NAMES[p.citizen],
            p.kind.name(),
            site + 1
        );
        self.kingdoms[civ].satisfaction = (self.kingdoms[civ].satisfaction + 4).min(100);
        self.event(text);
        Ok(())
    }
    pub fn set_tax(&mut self, tax: u8) {
        let tax = tax.min(40);
        if tax == self.kingdoms[0].tax {
            return;
        }
        self.kingdoms[0].tax = tax;
        for c in self.citizens.iter_mut().filter(|c| c.kingdom == 0) {
            c.memory =
                format!("Our ruler set income tax to {tax}%. I hope it funds something useful.");
        }
        self.event(format!(
            "You set income tax to {tax}%. It applies next season."
        ));
    }
    pub fn trade_treaty(&mut self, rival: usize) -> Result<(), &'static str> {
        if !(1..3).contains(&rival) {
            return Err("Choose a neighboring kingdom.");
        }
        if self.trade[rival] {
            return Err("A trade treaty is already active.");
        }
        if self.relations[rival] < 0 {
            return Err("Improve relations before proposing trade.");
        }
        if self.kingdoms[0].treasury < 30 {
            return Err("A trade mission needs 30 coins.");
        }
        self.kingdoms[0].treasury -= 30;
        self.trade[rival] = true;
        self.relations[rival] += 10;
        self.event(format!(
            "Trade opened with {}. Both kingdoms' citizens earn 6 extra coins each season.",
            self.kingdoms[rival].name
        ));
        Ok(())
    }
    pub fn goodwill(&mut self, rival: usize) -> Result<(), &'static str> {
        if !(1..3).contains(&rival) || self.kingdoms[0].treasury < 20 {
            return Err("A goodwill gift needs 20 coins and a neighboring kingdom.");
        }
        self.kingdoms[0].treasury -= 20;
        self.kingdoms[rival].treasury += 20;
        self.relations[rival] = (self.relations[rival] + 15).min(100);
        self.event(format!(
            "Your gift improved relations with {}.",
            self.kingdoms[rival].name
        ));
        Ok(())
    }
    pub fn request_land(&mut self, rival: usize) -> Result<(), &'static str> {
        if !(1..3).contains(&rival) {
            return Err("Choose a neighboring kingdom.");
        }
        if self.relations[rival] < 20 {
            return Err("This ruler wants better relations before selling land.");
        }
        let owned = self
            .parcels
            .iter()
            .filter(|s| s.owner == Some(rival))
            .count();
        if owned <= 2 {
            return Err("This ruler will not sell their last two parcels.");
        }
        let parcel = (0..16)
            .find(|&id| {
                self.parcels[id].owner == Some(rival)
                    && self.parcels[id].project.is_none()
                    && self.adjacent(0, id)
                    && self.connected_without(rival, id)
            })
            .ok_or("No undeveloped neighboring parcel is available for sale.")?;
        let price = self.land_price(parcel) + 80 - self.relations[rival].max(0).min(60) as u64;
        self.offer = Some(LandOffer {
            seller: rival,
            parcel,
            price,
            expires: self.season + 2,
        });
        Ok(())
    }
    fn connected_without(&self, civ: usize, excluded: usize) -> bool {
        let ids: Vec<_> = (0..16)
            .filter(|&i| i != excluded && self.parcels[i].owner == Some(civ))
            .collect();
        let Some(&first) = ids.first() else {
            return false;
        };
        let mut seen = vec![first];
        let mut n = 0;
        while n < seen.len() {
            let a = seen[n];
            for &b in &ids {
                if !seen.contains(&b) && (a % 4).abs_diff(b % 4) + (a / 4).abs_diff(b / 4) == 1 {
                    seen.push(b);
                }
            }
            n += 1;
        }
        seen.len() == ids.len()
    }
    pub fn accept_land(&mut self) -> Result<(), &'static str> {
        let o = self.offer.clone().ok_or("Request an offer first.")?;
        if self.season >= o.expires
            || self.parcels[o.parcel].owner != Some(o.seller)
            || self.parcels[o.parcel].project.is_some()
            || !self.adjacent(0, o.parcel)
            || !self.connected_without(o.seller, o.parcel)
        {
            self.offer = None;
            return Err("That offer is no longer available.");
        }
        if self.kingdoms[0].treasury < o.price {
            return Err("Not enough treasury funds for this offer.");
        }
        self.kingdoms[0].treasury -= o.price;
        self.kingdoms[o.seller].treasury += o.price;
        self.parcels[o.parcel].owner = Some(0);
        self.parcels[o.parcel].reserved = false;
        self.relations[o.seller] += 5;
        self.offer = None;
        self.event(format!(
            "You negotiated the purchase of parcel {} for {} coins.",
            o.parcel + 1,
            o.price
        ));
        Ok(())
    }
    pub fn wealth(&self, civ: usize) -> u64 {
        self.kingdoms[civ].treasury
            + self
                .citizens
                .iter()
                .filter(|c| c.kingdom == civ)
                .map(|c| c.savings)
                .sum::<u64>()
    }
    pub fn advance(&mut self, dt: f32) {
        if !dt.is_finite() || dt <= 0.0 {
            return;
        }
        self.elapsed += dt;
        while self.elapsed >= SEASON_SECONDS {
            self.elapsed -= SEASON_SECONDS;
            self.next_season();
        }
    }
    fn next_season(&mut self) {
        self.season += 1;
        for id in 0..self.projects.len() {
            let p = &mut self.projects[id];
            if p.parcel.is_some() && !p.completed {
                p.remaining -= 1;
                if p.remaining == 0 {
                    p.completed = true;
                    self.citizens[p.citizen].memory = format!(
                        "My {} is finished. Our ruler made this possible.",
                        p.kind.name()
                    );
                    let text = format!(
                        "{} completed a {}! {}.",
                        NAMES[p.citizen],
                        p.kind.name(),
                        p.kind.benefit()
                    );
                    self.event(text);
                }
            }
        }
        for civ in 0..3 {
            let completed: Vec<_> = self
                .projects
                .iter()
                .filter(|p| p.completed && self.citizens[p.citizen].kingdom == civ)
                .map(|p| p.kind)
                .collect();
            let count = |kind| completed.iter().filter(|&&k| k == kind).count() as u64;
            let trade = if civ == 0 {
                self.trade.iter().filter(|&&b| b).count() as u64
            } else {
                self.trade[civ] as u64
            };
            let k = &mut self.kingdoms[civ];
            let old_inventions = k.research / 10;
            k.research += count(ProjectKind::Observatory) * 2;
            let gross = 20 + count(ProjectKind::Workshop) * 5 + k.research / 10 * 4 + trade * 6;
            let gross = if k.satisfaction < 35 {
                gross * 3 / 4
            } else {
                gross
            };
            k.last_tax = 0;
            for c in self.citizens.iter_mut().filter(|c| c.kingdom == civ) {
                let tax = gross * k.tax as u64 / 100;
                c.savings += gross - tax;
                k.treasury += tax;
                k.last_tax += tax;
            }
            let target = (80 - k.tax as i32 * 2
                + completed.len() as i32 * 3
                + count(ProjectKind::Garden) as i32 * 3)
                .clamp(10, 100);
            k.satisfaction += (target - k.satisfaction).clamp(-4, 4);
            if k.research / 10 > old_inventions {
                let text = format!(
                    "{} discovered invention {}. Citizen earnings increased!",
                    k.name,
                    k.research / 10
                );
                self.event(text);
            }
        }
        if self
            .offer
            .as_ref()
            .is_some_and(|o| self.season >= o.expires)
        {
            self.offer = None;
        }
        for civ in 1..3 {
            self.kingdoms[civ].tax = if self.kingdoms[civ].satisfaction < 40 {
                10
            } else if civ == 1 {
                25
            } else {
                15
            };
            let expansion = (0..16).find(|&id| {
                self.parcels[id].owner.is_none()
                    && self.adjacent(civ, id)
                    && self.kingdoms[civ].treasury >= self.land_price(id)
            });
            if self.season % 3 == 0 {
                if let Some(id) = expansion {
                    let _ = self.buy_land(civ, id);
                    continue;
                }
            }
            let preferred = if civ == 1 {
                ProjectKind::Workshop
            } else {
                ProjectKind::Observatory
            };
            let candidate = (0..self.projects.len())
                .filter(|&p| {
                    self.citizens[self.projects[p].citizen].kingdom == civ
                        && self.projects[p].parcel.is_none()
                        && self.site_for(p).is_some()
                        && self.kingdoms[civ].treasury >= self.projects[p].kind.cost()
                })
                .min_by_key(|&p| self.projects[p].kind != preferred);
            if let Some(p) = candidate {
                let _ = self.fund(civ, p);
            } else if let Some(id) = (0..16).find(|&id| {
                self.parcels[id].owner.is_none()
                    && self.adjacent(civ, id)
                    && self.kingdoms[civ].treasury >= self.land_price(id)
            }) {
                let _ = self.buy_land(civ, id);
            }
        }
        // A completed ambition gives its citizen a new proposal, keeping audiences alive.
        if self.season % 3 == 0 {
            for citizen in 0..self.citizens.len() {
                if !self
                    .projects
                    .iter()
                    .any(|p| p.citizen == citizen && !p.completed)
                {
                    let kind = match (self.season as usize / 3 + citizen) % 3 {
                        0 => ProjectKind::Workshop,
                        1 => ProjectKind::Garden,
                        _ => ProjectKind::Observatory,
                    };
                    self.projects.push(Project {
                        citizen,
                        kind,
                        parcel: None,
                        remaining: 0,
                        completed: false,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn taxes_transfer_only_a_share_of_income() {
        let mut r = Realm::new(42);
        r.set_tax(25);
        let before = r.wealth(0);
        r.advance(SEASON_SECONDS);
        assert_eq!(r.kingdoms[0].last_tax, 20);
        assert_eq!(r.kingdoms[0].treasury, 200);
        assert_eq!(r.citizens[0].savings, 40);
        assert_eq!(r.wealth(0) - before, 80);
    }
    #[test]
    fn land_requires_money_adjacency_and_vacancy() {
        let mut r = Realm::new(42);
        let before = r.kingdoms[0].treasury;
        assert!(r.buy_land(0, 10).is_err());
        assert!(r.buy_land(0, 3).is_err());
        assert_eq!(r.kingdoms[0].treasury, before);
        r.buy_land(0, 4).unwrap();
        assert_eq!(r.parcels[4].owner, Some(0));
        assert!(r.buy_land(0, 4).is_err());
        r.kingdoms[0].treasury = 0;
        assert!(r.buy_land(0, 5).is_err());
    }
    #[test]
    fn grants_respect_reservations_ownership_and_complete_over_time() {
        let mut r = Realm::new(42);
        r.parcels[0].reserved = true;
        assert!(r.fund(0, 0).is_err());
        assert!(r.fund(0, 4).is_err());
        r.parcels[0].reserved = false;
        r.fund(0, 0).unwrap();
        assert!(r.fund(0, 0).is_err());
        r.advance(SEASON_SECONDS);
        assert!(!r.projects[0].completed);
        r.advance(SEASON_SECONDS);
        assert!(r.projects[0].completed);
        assert!(r.citizens[0].memory.contains("finished"));
        assert_eq!(r.citizens[0].savings, 64); // 17 + 22 net coins, workshop completed
    }
    #[test]
    fn treaties_are_bilateral_and_cannot_be_bought_twice() {
        let mut r = Realm::new(1);
        r.trade_treaty(1).unwrap();
        let t = r.kingdoms[0].treasury;
        assert!(r.trade_treaty(1).is_err());
        assert_eq!(r.kingdoms[0].treasury, t);
        r.advance(SEASON_SECONDS);
        assert_eq!(r.citizens[0].savings, 48);
        assert_eq!(r.citizens[4].savings, 48);
    }
    #[test]
    fn diplomacy_transfers_funds_and_revalidates_offer() {
        let mut r = Realm::new(1);
        r.parcels[2].owner = Some(1);
        r.request_land(1).unwrap();
        let price = r.offer.as_ref().unwrap().price;
        let total = r.kingdoms[0].treasury + r.kingdoms[1].treasury;
        r.accept_land().unwrap();
        assert_eq!(r.parcels[2].owner, Some(0));
        assert_eq!(r.kingdoms[0].treasury, 180 - price);
        assert_eq!(r.kingdoms[0].treasury + r.kingdoms[1].treasury, total);
        assert!(r.accept_land().is_err());
        let mut r = Realm::new(1);
        r.parcels[2].owner = Some(1);
        r.request_land(1).unwrap();
        r.season += 2;
        assert!(r.accept_land().is_err());
        assert_eq!(r.parcels[2].owner, Some(1));
    }
    #[test]
    fn rivals_progress_and_time_is_deterministic() {
        let mut a = Realm::new(19);
        let mut b = Realm::new(19);
        for _ in 0..120 {
            a.advance(SEASON_SECONDS / 2.0);
        }
        b.advance(SEASON_SECONDS * 60.0);
        assert_eq!(a.wealth(1), b.wealth(1));
        assert!(
            a.projects
                .iter()
                .any(|p| p.completed && a.citizens[p.citizen].kingdom == 1)
        );
        assert!(a.parcels.iter().filter(|p| p.owner == Some(1)).count() > 2);
        assert!(
            a.kingdoms
                .iter()
                .all(|k| (0..=100).contains(&k.satisfaction))
        );
    }

    #[test]
    fn completed_sites_can_be_improved_without_double_booking_construction() {
        let mut r = Realm::new(42);
        r.fund(0, 0).unwrap();
        assert!(r.fund(0, 2).is_err());
        r.advance(SEASON_SECONDS * 2.0);
        r.kingdoms[0].treasury = 200;
        r.fund(0, 2).unwrap();
        assert_eq!(r.projects[2].parcel, Some(0));
        r.advance(SEASON_SECONDS * 2.0);
        assert!(r.projects[0].completed && r.projects[2].completed);
        assert_eq!(r.parcels[0].project, Some(2));
    }

    #[test]
    fn high_taxes_reduce_satisfaction_and_research_creates_inventions() {
        let mut r = Realm::new(42);
        r.set_tax(255);
        assert_eq!(r.kingdoms[0].tax, 40);
        r.advance(SEASON_SECONDS * 10.0);
        assert!(r.kingdoms[0].satisfaction < 35);
        let before = r.citizens[0].savings;
        r.advance(SEASON_SECONDS);
        assert_eq!(r.citizens[0].savings - before, 9);
        let mut r = Realm::new(2);
        r.fund(0, 0).unwrap();
        r.advance(SEASON_SECONDS * 6.0);
        assert_eq!(r.kingdoms[0].research, 10);
        assert!(
            r.events
                .iter()
                .any(|e| e.contains("Brookhaven") && e.contains("invention"))
        );
    }

    #[test]
    fn map_coordinates_match_every_parcel_and_reject_outside_world() {
        for id in 0..16 {
            let (x, z) = Realm::center(id);
            assert_eq!(Realm::parcel_at(x, z), Some(id));
        }
        assert_eq!(Realm::parcel_at(30.0, 0.0), None);
        assert_eq!(Realm::parcel_at(f32::NAN, 0.0), None);
    }
}
