use three_d::egui::{self, Color32, RichText};
use yumon_pet::kingdom::{COLORS, NAMES, Realm, SEASON_SECONDS};

fn report(result: Result<(), &'static str>, notice: &mut String, success: &str) {
    *notice = result
        .map(|_| success.to_string())
        .unwrap_or_else(str::to_string);
}

pub fn panel(
    ui: &mut egui::Ui,
    realm: &mut Realm,
    selected: &mut usize,
    parcel: &mut usize,
    notice: &mut String,
) {
    ui.heading("Your kingdom");
    let k = &realm.kingdoms[0];
    ui.label(format!(
        "Season {} · Treasury {} coins",
        realm.season, k.treasury
    ));
    ui.label(format!(
        "Satisfaction {}% · Research {} · Inventions {}",
        k.satisfaction,
        k.research,
        k.research / 10
    ));
    ui.small(format!(
        "Last collection: {} coins · Next in {:.0}s",
        k.last_tax,
        SEASON_SECONDS - realm.elapsed
    ));
    let mut tax = k.tax;
    if ui
        .add(egui::Slider::new(&mut tax, 0..=40).text("Income tax %"))
        .changed()
    {
        realm.set_tax(tax);
    }
    ui.small("Taxes come from citizen earnings. Satisfaction below 35% reduces earnings. A season lasts 25 seconds.");
    if !notice.is_empty() {
        ui.label(RichText::new(notice.as_str()).color(Color32::LIGHT_YELLOW));
    }

    egui::CollapsingHeader::new("Royal audience")
        .default_open(true)
        .show(ui, |ui| {
            let pending: Vec<_> = realm
                .projects
                .iter()
                .enumerate()
                .filter(|(_, p)| realm.citizens[p.citizen].kingdom == 0 && !p.completed)
                .map(|(id, _)| id)
                .collect();
            if pending.is_empty() {
                ui.label("Your projects are complete. New ambitions arrive every three seasons.");
            }
            for id in pending {
                let p = realm.projects[id].clone();
                ui.group(|ui| {
                    if ui
                        .selectable_label(
                            *selected == p.citizen,
                            format!("{}: {}", NAMES[p.citizen], p.kind.name()),
                        )
                        .clicked()
                    {
                        *selected = p.citizen;
                    }
                    ui.small(p.kind.benefit());
                    if let Some(site) = p.parcel {
                        ui.label(format!(
                            "Building on parcel {} · {} seasons left",
                            site + 1,
                            p.remaining
                        ));
                    } else {
                        let site = realm.site_for(id);
                        ui.small(
                            site.map(|s| {
                                format!(
                                    "{}: parcel {}",
                                    if realm.parcels[s].project.is_some() {
                                        "Improve existing site"
                                    } else {
                                        "Proposed site"
                                    },
                                    s + 1
                                )
                            })
                            .unwrap_or_else(|| {
                                "Needs matching, unreserved land. Check Territory.".into()
                            }),
                        );
                        if ui
                            .add_enabled(
                                site.is_some() && realm.kingdoms[0].treasury >= p.kind.cost(),
                                egui::Button::new(format!("Fund · {} coins", p.kind.cost())),
                            )
                            .clicked()
                        {
                            report(
                                realm.fund(0, id),
                                notice,
                                "Project approved. Construction has begun.",
                            );
                        }
                    }
                });
            }
        });
    egui::CollapsingHeader::new("Territory")
        .default_open(true)
        .show(ui, |ui| {
            ui.small("Blue: yours · Gold / purple: rivals · Gray: unclaimed");
            ui.small("The selected parcel has a pale border in the scene.");
            egui::Grid::new("parcel_map")
                .spacing([4.0, 4.0])
                .show(ui, |ui| {
                    for id in 0..16 {
                        let s = &realm.parcels[id];
                        let c = s.owner.map(|i| COLORS[i]).unwrap_or([85, 95, 105]);
                        let label = format!(
                            "{}{}",
                            id + 1,
                            if s.project.is_some() {
                                " •"
                            } else if s.reserved {
                                " R"
                            } else {
                                ""
                            }
                        );
                        let button = egui::Button::new(RichText::new(label).color(Color32::WHITE))
                            .fill(Color32::from_rgb(c[0] / 2, c[1] / 2, c[2] / 2))
                            .stroke(egui::Stroke::new(
                                if id == *parcel { 2.0 } else { 0.0 },
                                Color32::WHITE,
                            ))
                            .min_size(egui::vec2(58.0, 28.0));
                        if ui.add(button).clicked() {
                            *parcel = id;
                        }
                        if id % 4 == 3 {
                            ui.end_row();
                        }
                    }
                });
            let s = realm.parcels[*parcel].clone();
            ui.label(format!(
                "Parcel {} · Suitable for {}",
                *parcel + 1,
                s.kind.name()
            ));
            ui.small(format!(
                "Owner: {}",
                s.owner
                    .map(|i| realm.kingdoms[i].name)
                    .unwrap_or("Unclaimed")
            ));
            if let Some(project) = s.project {
                let p = &realm.projects[project];
                ui.label(format!(
                    "{}'s {} · {}",
                    NAMES[p.citizen],
                    p.kind.name(),
                    if p.completed {
                        "complete"
                    } else {
                        "under construction"
                    }
                ));
            } else if s.owner == Some(0) {
                ui.checkbox(
                    &mut realm.parcels[*parcel].reserved,
                    "Reserve for future public use",
                );
            } else if s.owner.is_none() {
                let price = realm.land_price(*parcel);
                if ui
                    .add_enabled(
                        realm.adjacent(0, *parcel) && realm.kingdoms[0].treasury >= price,
                        egui::Button::new(format!("Purchase · {price} coins")),
                    )
                    .clicked()
                {
                    report(
                        realm.buy_land(0, *parcel),
                        notice,
                        "Territory acquired. Citizens can propose projects here.",
                    );
                }
                if !realm.adjacent(0, *parcel) {
                    ui.small("You can only buy land sharing a border with your kingdom.");
                }
            } else {
                ui.small("Ask this kingdom for a land offer under Diplomacy.");
            }
        });
    egui::CollapsingHeader::new("Diplomacy").show(ui, |ui| {
        for rival in 1..3 {
            ui.group(|ui| {
                ui.strong(realm.kingdoms[rival].name);
                ui.label(format!(
                    "Relations: {} · {}",
                    realm.relations[rival],
                    if realm.trade[rival] {
                        "Trade treaty active"
                    } else {
                        "No trade treaty"
                    }
                ));
                if ui
                    .add_enabled(
                        !realm.trade[rival],
                        egui::Button::new("Open trade · 30 coins"),
                    )
                    .on_hover_text("Both kingdoms gain +6 earnings per citizen each season.")
                    .clicked()
                {
                    report(realm.trade_treaty(rival), notice, "Trade treaty signed.");
                }
                if ui
                    .button("Goodwill gift · 20 coins")
                    .on_hover_text("Transfer 20 coins to this kingdom; relations improve by 15.")
                    .clicked()
                {
                    report(
                        realm.goodwill(rival),
                        notice,
                        "Your gift was well received.",
                    );
                }
                if ui.button("Negotiate a land purchase").clicked() {
                    report(
                        realm.request_land(rival),
                        notice,
                        "A land offer is ready for your decision.",
                    );
                }
            });
        }
        if let Some(o) = realm.offer.clone() {
            ui.label(format!(
                "{} offers parcel {} for {} coins. Expires in season {}.",
                realm.kingdoms[o.seller].name,
                o.parcel + 1,
                o.price,
                o.expires
            ));
            if ui.button("Accept land offer").clicked() {
                report(realm.accept_land(), notice, "Land agreement signed.");
            }
            if ui.button("Decline offer").clicked() {
                realm.offer = None;
                *notice = "Offer declined.".into();
            }
        }
    });
    egui::CollapsingHeader::new("Civilization standings").show(ui, |ui| {
        ui.small(
            "Wealth includes the treasury and citizen savings. Research and wellbeing count too.",
        );
        let mut ids = [0, 1, 2];
        ids.sort_by_key(|&c| std::cmp::Reverse(realm.wealth(c)));
        for (rank, civ) in ids.into_iter().enumerate() {
            ui.label(format!(
                "{}. {} · {} coins",
                rank + 1,
                realm.kingdoms[civ].name,
                realm.wealth(civ)
            ));
            ui.small(format!(
                "{} parcels · {} inventions · {}% satisfaction",
                realm
                    .parcels
                    .iter()
                    .filter(|s| s.owner == Some(civ))
                    .count(),
                realm.kingdoms[civ].research / 10,
                realm.kingdoms[civ].satisfaction
            ));
        }
    });
    egui::CollapsingHeader::new("Kingdom chronicle").show(ui, |ui| {
        for e in &realm.events {
            ui.label(e);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ruler_panel_renders_initial_and_developed_kingdoms() {
        let ctx = egui::Context::default();
        let mut realm = Realm::new(42);
        let (mut selected, mut parcel, mut notice) = (0, 0, String::new());
        for developed in [false, true] {
            if developed {
                realm.fund(0, 0).unwrap();
                realm.advance(SEASON_SECONDS * 8.0);
            }
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1200.0, 900.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::SidePanel::right("test_court").show(ctx, |ui| {
                        panel(ui, &mut realm, &mut selected, &mut parcel, &mut notice);
                    });
                },
            );
            assert!(!output.shapes.is_empty());
        }
    }
}
