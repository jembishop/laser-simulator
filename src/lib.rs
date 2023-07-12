use std::collections::VecDeque;
use std::f32::consts::PI;

use egui::plot::{Line, Plot, PlotPoints};
use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Stroke};

use glam::Vec2;
use ndarray::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::Rng;
use rand::{
    distributions::{Bernoulli, Uniform},
    prelude::Distribution,
    rngs::ThreadRng,
    thread_rng,
};
use rand_distr::Poisson;

#[derive(Copy, Clone)]
enum Level {
    Ground,
    Uncharged,
    Charged,
    Pumped,
}

#[derive(Copy, Clone, Debug, TryFromPrimitive, PartialEq, Eq, IntoPrimitive)]
#[repr(u8)]
enum Levels {
    Two,
    Three,
    Four,
}

#[derive(Copy, Clone)]
struct Atom {
    level: Level,
}

#[derive(Copy, Clone, Debug, TryFromPrimitive, PartialEq, Eq, IntoPrimitive)]
#[repr(u8)]
enum PhotonType {
    Pump,
    Stim,
    Spont,
}

struct Photon {
    pos: Vec2,
    vel: Vec2,
    ty: PhotonType,
    escaped: bool,
}

pub struct App {
    atoms: Array2<Atom>,
    photons: Vec<Photon>,
    iter: usize,
    rng: ThreadRng,

    levels: Levels,
    spont_chance: f32,
    reflect_chance: f32,
    interact_chance: f32,
    pump_rate: f32,
    pump_drop_chance: f32,
    decharged_drop_chance: f32,
    q: f32,
    dt: f32,
    show_pump_photons: bool,

    t: f32,
    level_pop_buffer: VecDeque<[f32; 4]>,
    time_buffer: VecDeque<f32>,
    escaped_buffer: VecDeque<f32>,

    show_graph: bool,
    laser_zone: Vec2,
    diagram_rand_dir: Vec2,
    stim_col: usize,
    t_anim: f32,
}

const N_ATOMS_X: usize = 88;
const N_ATOMS_Y: usize = 24;
const N_ATOMS_PAD: usize = 12;

pub const SCREEN_SIZE: [u32; 2] = [1080, 720];

const PHOTON_SPEED: f32 = 1.;

const MIRROR_WIDTH: f32 = 10.;
const VIEW_ASPECT: f32 = 6.;

const ANGLE_PAD: f32 = 1.;
const LEVEL_COLORS: [Color32; 4] = [
    Color32::DARK_GRAY,
    Color32::DARK_RED,
    Color32::RED,
    Color32::GOLD,
];
const PHOTON_COLORS: [Color32; 3] = [Color32::WHITE, Color32::GREEN, Color32::YELLOW];

const REFLECT_Y_NOISE: f64 = 0.5;
const DT_ANIM: f32 = 0.008;

const MAX_PUMP: f32 = 400.;

const BUFFER_SIZE: usize = 2048;

const AVG_WINDOW: usize = 256;

fn two_level_text(ui: &mut egui::Ui, t: f32, rand_dir: Vec2, stim_col: usize) {
    // ui.spacing_mut().item_spacing = egui::Vec2::new(0., 0.);
    ui.heading("Welcome to laser simulator!");
    ui.horizontal_wrapped(|ui| {
        ui.label(
            "To explore the workings of a laser, we must first become familiar with the mechanism of stimulated emission.\
             The simulation currently shows a two level 'laser'. This laser has atoms which can be in two states,"
        );
        ui.label(RichText::new("charged").color(LEVEL_COLORS[2]));
        ui.label("or");
        ui.label(RichText::new("ground").color(LEVEL_COLORS[0]));
        ui.label(
                ". The 'pump' which is located on the top and bottom of the laser provides a steady stream of photons, which when absorbed by ground state atoms, will cause the atom be excited to the charged state. \
                 Once in this state, the atom will eventually drop back to the ground state by one of two processes. Either the atom will randomly emit a photon and drop back to the ground state, \
                 or if a photon happens to be passing by it will emit a photon in the same direction as the nearby photon. The former process is called spontaneous emission and the latter process is called stimulated emission, which puts the SE in LASER!. \
                 In the two level laser the pump and emitted photons are the same energy, though in the simulation we colour the photons according to their source:\n("
                );
        ui.label(RichText::new("pump,").color(PHOTON_COLORS[0]));
        ui.label(RichText::new("spontaneous,").color(PHOTON_COLORS[1]));
        ui.label(RichText::new("stimulated").color(PHOTON_COLORS[2]));
        ui.label(").You will notice the laser output doesn't look very good at the moment, try putting the laser in the 3 level state using the dropdown on the left, to see more info.");
    });
    // ui.debug_paint_cursor();
    ui.set_min_height(100.);
    let (r, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::hover());
    let rect = r.rect;
    let wid = rect.width();
    let delt = 1.2 * wid / 4.;
    let offset = rect.left_top() + egui::Vec2::new(delt - 80., 30.);
    let offset2 = offset + egui::Vec2::new(delt, 0.);
    let offset3 = offset2 + egui::Vec2::new(delt, 0.);
    let phot_dest = 80.;

    let t2 = 2. * t;
    let rad1 = 12.;
    let rad2 = rad1 / 2.;
    let dest2 = rand_dir * phot_dest / 2.;
    let stim_col = PHOTON_COLORS[stim_col];

    let (col1, col2) = if t2 < 1. {
        let x0 = egui::Vec2::new(-phot_dest, 0.);
        let x = x0 * (1. - t2);
        // absorbed photon
        painter.circle(offset + x, rad2, stim_col, Stroke::new(0., stim_col));
        (LEVEL_COLORS[0], LEVEL_COLORS[2])
    } else {
        let col = PHOTON_COLORS[2];
        let x2 = dest2.perp() * (t2 - 1.);
        let x2 = egui::Vec2::new(x2.x, x2.y);
        // spont photon
        painter.circle(offset2 + x2, rad2, col, Stroke::new(0., col));

        // stim photon
        let x3 = dest2 * (t2 - 1.);
        let x3 = egui::Vec2::new(x3.x, x3.y);
        painter.circle(offset3 + x3, rad2, stim_col, Stroke::new(0., col));

        (LEVEL_COLORS[2], LEVEL_COLORS[0])
    };

    let paint_text = |text: &str, off: egui::Pos2| {
        let offset = off + egui::Vec2::new(0., 50.);
        painter.text(
            offset,
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(12.),
            Color32::GRAY,
        );
    };
    // atoms
    painter.circle(offset, rad1, col1, Stroke::new(0., col1));
    paint_text("Absorption", offset);
    painter.circle(offset2, rad1, col2, Stroke::new(0., col2));
    paint_text("Spontaneous Emission", offset2);
    painter.circle(offset3, rad1, col2, Stroke::new(0., col2));
    paint_text("Stimulated Emission", offset3);
    // stim original photon
    let x4_g = dest2 * (t2 - 1.);
    let perp = rand_dir.perp();
    let x4_p = x4_g + 16. * perp;
    let perp = egui::Vec2::new(x4_p.x, x4_p.y);
    painter.circle(offset3 + perp, rad2, stim_col, Stroke::new(0., stim_col));
}

fn three_level_text(ui: &mut egui::Ui, t: f32, rand_dir: Vec2) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            "You may have noticed in the previous example that the laser didn't seem to be doing much \
            in fact it's impossible to get a two level laser to actually work! To see why we need to understand \
            what makes a laser work in the first place. A laser requires a higher proportion of the atoms in the \
            charged state than the ground state. Once this is condition is achieved (known as population inversion) \
            a photon is more likely to cause a stimulated emission rather than be absorbed when travelling through the \
            material. This new photon can stimulate other emission leading to a big amplication in the number of photons!\
            \n\nSo why didn't the two level laser work? This setup could never achieve a population inversion \
            no matter how high we set the pump rate, because the incoming pump photons would always spoil any population \
            inversion by causing the charged atoms to drop to the ground state by stimulated emission. To solve this we must prevent \
            the pump photons from interfering with our charged atoms, while ensuring the pump photons still charge up the \
            atoms in the ground state. To do this we introduce another level above the charged state called the");
        ui.label(RichText::new("pumped").color(LEVEL_COLORS[3]));
        ui.label(
            "state . The pump photons raise \
            the atoms to this 'pumped' state, and then the atom quickly jumps down to the charged state via dissipating the energy \
            by some mechanism. Crucially the pump photons do not interact with the charged atoms as they have the wrong energy.\
            The laser should be 'lasing' now!"
        );
    });
    ui.set_min_height(100.);
    let (r, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::hover());
    let rect = r.rect;
    let wid = rect.width();
    let delt = 1.2 * wid / 4.;
    let offset = rect.left_top() + egui::Vec2::new(delt - 80., 30.);
    let offset2 = offset + egui::Vec2::new(delt, 0.);
    let offset3 = offset2 + egui::Vec2::new(delt, 0.);
    let phot_dest = 80.;

    let t2 = 2. * t;
    let rad1 = 12.;
    let rad2 = rad1 / 2.;
    let dest2 = rand_dir * phot_dest / 2.;
    let (col1, col2) = if t2 < 1. {
        let x0 = egui::Vec2::new(-phot_dest, 0.);
        let x = x0 * (1. - t2);
        // absorbed photon
        let col = PHOTON_COLORS[0];
        painter.circle(offset + x, rad2, col, Stroke::new(0., col));
        (LEVEL_COLORS[0], LEVEL_COLORS[3])
    } else {
        (LEVEL_COLORS[3], LEVEL_COLORS[2])
    };

    let paint_text = |text: &str, off: egui::Pos2| {
        let offset = off + egui::Vec2::new(0., 50.);
        painter.text(
            offset,
            Align2::CENTER_CENTER,
            text,
            FontId::proportional(12.),
            Color32::GRAY,
        );
    };
    // atoms
    painter.circle(offset, rad1, col1, Stroke::new(0., col1));
    paint_text("Pump photon absorption", offset);
    painter.circle(offset2, rad1, col2, Stroke::new(0., col2));
    paint_text("Pump drop to charged", offset2);
    painter.circle(offset3, rad1, LEVEL_COLORS[2], Stroke::new(0., col2));
    paint_text("Pump photon doesn't\ninteract with charged state", offset3);
    // stim original photon
    let x4_g = dest2 * (t2 - 1.);
    let perp = rand_dir.perp();
    let x4_p = x4_g + 16. * perp;
    let perp = egui::Vec2::new(x4_p.x, x4_p.y);
    painter.circle(offset3 + perp, rad2, PHOTON_COLORS[0], Stroke::new(0., col1));
}

impl Default for App {
    fn default() -> Self {
        let atoms = Array2::from_elem(
            (N_ATOMS_X, N_ATOMS_Y),
            Atom {
                level: Level::Ground,
            },
        );
        let photons = Vec::new();
        let rng = thread_rng();
        let levels = Levels::Two;

        let level_pop_buffer = VecDeque::new();
        let time_buffer = VecDeque::new();
        let escaped_buffer = VecDeque::from_iter(std::iter::repeat(0.).take(AVG_WINDOW));

        Self {
            atoms,
            photons,
            iter: 0,
            t: 0.,
            levels,
            rng,
            dt: 0.2,
            q: 0.98,
            show_pump_photons: true,
            pump_rate: 100.,
            spont_chance: 0.02,
            reflect_chance: 0.8,
            interact_chance: 0.1,
            pump_drop_chance: 0.3,
            decharged_drop_chance: 0.4,

            level_pop_buffer,
            time_buffer,
            escaped_buffer,

            show_graph: false,
            laser_zone: Vec2::ZERO,
            diagram_rand_dir: Vec2::ONE,
            t_anim: 0.,
            stim_col: 0,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        let dt = self.dt;
        let new_dist = Poisson::new(f32::max(self.pump_rate * dt, 0.01)).unwrap();
        let angle_dist = Uniform::new(ANGLE_PAD, PI - ANGLE_PAD);
        let angle_dist2 = Uniform::new(0., 2. * PI);
        let len_dist = Uniform::new(0., N_ATOMS_X as f32);
        let interact_dist = Bernoulli::new((self.interact_chance * dt) as f64).unwrap();
        let spon_dist = Bernoulli::new((self.spont_chance as f64) * (dt as f64)).unwrap();
        let pump_drop_dist = Bernoulli::new((self.pump_drop_chance * dt) as f64).unwrap();
        let decharge_drop_dist = Bernoulli::new((self.decharged_drop_chance * dt) as f64).unwrap();
        let reflect_dist = Bernoulli::new(self.reflect_chance as f64).unwrap();
        let reflect_noise = Uniform::new(-REFLECT_Y_NOISE, REFLECT_Y_NOISE);
        let q_dist = Bernoulli::new((dt * (1. - self.q) / 10.) as f64).unwrap();

        let pop_counts = self.atoms.iter().fold([0; 4], |mut cum, x| {
            cum[x.level as usize] += 1;
            cum
        });
        let pop_counts_f = pop_counts.map(|x| (x as f32) / (self.atoms.len() as f32));
        egui::CentralPanel::default().show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.style_mut().spacing.slider_width = 400.0;
                    ui.label("Pump Rate:");
                    ui.add(egui::Slider::new(
                        &mut self.pump_rate,
                        std::ops::RangeInclusive::new(0.1, MAX_PUMP),
                    ));
                    ui.label("Spontaneous Emission Probability:");
                    ui.add(egui::Slider::new(
                        &mut self.spont_chance,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.label("Mirror Reflection Probability:");
                    ui.add(egui::Slider::new(
                        &mut self.reflect_chance,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.label("dt");
                    ui.add(egui::Slider::new(
                        &mut self.dt,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.label("Q");
                    ui.add(egui::Slider::new(
                        &mut self.q,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.label("Photon Interaction Probability:");
                    ui.add(egui::Slider::new(
                        &mut self.interact_chance,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.label("Pumped -> Charged Probability:");
                    ui.add(egui::Slider::new(
                        &mut self.pump_drop_chance,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.label("Decharged -> Ground Probability:");
                    ui.add(egui::Slider::new(
                        &mut self.decharged_drop_chance,
                        std::ops::RangeInclusive::new(0., 1.),
                    ));

                    ui.horizontal(|ui| {
                        ui.checkbox(&mut self.show_pump_photons, "Show Pump Photons");
                        egui::ComboBox::from_label("")
                            .selected_text(format!("{:?} Levels", self.levels))
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.levels, Levels::Two, "Two Level");
                                ui.selectable_value(&mut self.levels, Levels::Three, "Three Level");
                                ui.selectable_value(&mut self.levels, Levels::Four, "Four Level");
                            });
                    });
                });
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        if ui.selectable_label(!self.show_graph, "Info").clicked() {
                            self.show_graph = false;
                        }
                        if ui.selectable_label(self.show_graph, "Graphs").clicked() {
                            self.show_graph = true;
                        }
                    });
                    if self.show_graph {
                        let lines = (0..4).map(|i| {
                            let vs: PlotPoints = self
                                .level_pop_buffer
                                .iter()
                                .map(|x| x[i])
                                .zip(self.time_buffer.iter())
                                .map(|(y, x)| [*x as f64, y as f64])
                                .collect();
                            let col = LEVEL_COLORS[i];
                            let line = Line::new(vs).stroke((2., col));
                            line
                        });

                        ui.label("Level Occupation");
                        Plot::new("Pop prop")
                            .view_aspect(VIEW_ASPECT)
                            .link_axis("g1", true, false)
                            .include_y(1.0)
                            .include_y(0.)
                            .show(ui, |plot_ui| {
                                for l in lines {
                                    plot_ui.line(l)
                                }
                            });

                        ui.label("Avg Power");
                        let escaped_mv_vec: Vec<f32> =
                            self.escaped_buffer.iter().copied().collect();
                        let escaped_mv_avg: Vec<f32> = escaped_mv_vec
                            .windows(AVG_WINDOW + 1)
                            .map(|x| {
                                x.iter().sum::<f32>()
                                    / ((AVG_WINDOW as f32) * f32::max(self.dt, 0.001))
                            })
                            .collect();

                        debug_assert!(escaped_mv_avg.len() == self.time_buffer.len());

                        let escaped: PlotPoints = escaped_mv_avg
                            .iter()
                            .zip(self.time_buffer.iter())
                            .map(|(y, x)| [*x as f64, *y as f64])
                            .collect();
                        let col = Color32::GREEN;
                        let line = Line::new(escaped).stroke((2., col));
                        Plot::new("Intensity")
                            .view_aspect(VIEW_ASPECT)
                            .link_axis("g1", true, false)
                            .include_y(0.)
                            .show(ui, |plot_ui| {
                                plot_ui.line(line);
                            });
                    } else {
                        if self.t_anim + DT_ANIM > 1. {
                            self.t_anim = 0.;
                            self.diagram_rand_dir =
                                Vec2::from_angle(angle_dist2.sample(&mut self.rng));
                            self.stim_col = self.rng.gen::<usize>() % 3;
                        } else {
                            self.t_anim += DT_ANIM;
                        }
                        match self.levels {
                            Levels::Two => {
                                two_level_text(
                                    ui,
                                    self.t_anim,
                                    self.diagram_rand_dir,
                                    self.stim_col,
                                );
                            }
                            Levels::Three => {
                                three_level_text(
                                    ui,
                                    self.t_anim,
                                    self.diagram_rand_dir,
                                );
                            }
                            _ => {}
                        }
                    }
                });
            });

            // Atoms
            // ui.debug_paint_cursor();
            let (r, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::hover());
            self.laser_zone = Vec2::new(r.rect.width(), r.rect.height());
            let atom_size = f32::min(
                self.laser_zone.x / ((N_ATOMS_X + N_ATOMS_PAD) as f32),
                self.laser_zone.y / ((N_ATOMS_Y + N_ATOMS_PAD) as f32),
            );
            let rt = r.rect.left_top();
            let pad = Vec2::new(0., 0.1 * self.laser_zone.y);
            let draw_off = Vec2::new(rt.x, rt.y) + pad;
            for ((i, j), a) in self.atoms.indexed_iter() {
                let x = i as f32;
                let y = j as f32;
                let v = Vec2::new(x, y) * atom_size;
                let col = LEVEL_COLORS[a.level as usize];
                let draw_v = v + draw_off + atom_size / 2.;
                let loc = Pos2::new(draw_v.x, draw_v.y);
                painter.circle(loc, atom_size / 2., col, Stroke::new(0., Color32::BLACK));
            }

            // Photons
            for p in &self.photons {
                let col = PHOTON_COLORS[p.ty as usize];

                let draw_v = p.pos * atom_size + draw_off;
                let loc = Pos2::new(draw_v.x, draw_v.y);
                if !(p.ty == PhotonType::Pump && !self.show_pump_photons) {
                    painter.circle(loc, atom_size / 4., col, Stroke::new(0., Color32::BLACK));
                }
            }

            // Mirror
            for left in [true, false] {
                let (x, col) = if left {
                    (0., Color32::WHITE)
                } else {
                    (
                        (N_ATOMS_X as f32) * atom_size,
                        Color32::WHITE.linear_multiply(self.reflect_chance + 0.1),
                    )
                };
                let min = Vec2::new(x - MIRROR_WIDTH / 2., 0.);
                let max = Vec2::new(x + MIRROR_WIDTH / 2., (N_ATOMS_Y as f32) * atom_size);
                let min = min + draw_off;
                let max = max + draw_off;
                let min = Pos2::new(min.x, min.y);
                let max = Pos2::new(max.x, max.y);
                painter.rect_filled(Rect { min, max }, 3., col)
            }
        });

        if dt == 0. {
            return;
        }

        self.level_pop_buffer.push_front(pop_counts_f);
        self.time_buffer.push_front(self.t);
        if self.level_pop_buffer.len() > BUFFER_SIZE {
            self.level_pop_buffer.pop_back();
            self.time_buffer.pop_back();
        }

        self.escaped_buffer.push_front(0.);
        if self.escaped_buffer.len() > BUFFER_SIZE + AVG_WINDOW {
            self.escaped_buffer.pop_back();
        }

        for _ in 0..4 {
            self.t += dt;
            let new = new_dist.sample(&mut self.rng) as usize;
            for up in [true, false] {
                for _ in 0..new / 2 {
                    let (y, dir) = if up {
                        (N_ATOMS_Y as f32, 1.)
                    } else {
                        (0., -1.)
                    };
                    let angle = angle_dist.sample(&mut self.rng);
                    let len = len_dist.sample(&mut self.rng);
                    let p = Photon {
                        pos: Vec2::new(len, y),
                        vel: -PHOTON_SPEED * Vec2::from_angle(angle) * dir,
                        ty: PhotonType::Pump,
                        escaped: false,
                    };
                    self.photons.push(p);
                }
            }
            let mut i = 0;
            while i < self.photons.len() {
                if self.photons[i].pos.x > (N_ATOMS_X + N_ATOMS_PAD) as f32
                    || self.photons[i].pos.y > (N_ATOMS_Y as f32) + 1.
                    || self.photons[i].pos.y < -1.
                {
                    self.photons.swap_remove(i);
                    i += 1;
                    continue;
                }
                let v = self.photons[i].pos;
                if v.x < N_ATOMS_X as f32 && v.y < N_ATOMS_Y as f32 && v.x > 0. && v.y > 0. {
                    let idx = (v.x as usize, v.y as usize);
                    let a = &mut self.atoms[idx];
                    if q_dist.sample(&mut self.rng) {
                        self.photons.swap_remove(i);
                        i += 1;
                        continue;
                    }
                    if interact_dist.sample(&mut self.rng) {
                        {
                            let p = &self.photons[i];
                            match a.level {
                                Level::Charged => {
                                    let emit = match self.levels {
                                        Levels::Four => {
                                            if p.ty != PhotonType::Pump {
                                                a.level = Level::Uncharged;
                                                true
                                            } else {
                                                false
                                            }
                                        }
                                        Levels::Three => {
                                            if p.ty != PhotonType::Pump {
                                                a.level = Level::Ground;
                                                true
                                            } else {
                                                false
                                            }
                                        }

                                        Levels::Two => {
                                            a.level = Level::Ground;
                                            true
                                        }
                                    };
                                    if emit {
                                        let p = Photon {
                                            pos: p.pos
                                                + 2. * Vec2::from_angle(
                                                    angle_dist2.sample(&mut self.rng),
                                                ),
                                            vel: p.vel.normalize(),
                                            ty: PhotonType::Stim,
                                            escaped: false,
                                        };
                                        self.photons.push(p);
                                    }
                                }
                                Level::Uncharged => {
                                    if matches!(self.levels, Levels::Four) {
                                        self.photons.swap_remove(i);
                                        a.level = Level::Charged;
                                    }
                                }
                                Level::Ground => {
                                    // level 2 always charge
                                    // level 3 blue pump, normal charge
                                    // level 4 only blue pump
                                    match self.levels {
                                        Levels::Two => {
                                            a.level = Level::Charged;
                                            self.photons.swap_remove(i);
                                        }
                                        Levels::Three => {
                                            if self.photons[i].ty == PhotonType::Pump {
                                                a.level = Level::Pumped;
                                            } else {
                                                a.level = Level::Charged;
                                            }
                                            self.photons.swap_remove(i);
                                        }
                                        Levels::Four => {
                                            if self.photons[i].ty == PhotonType::Pump {
                                                a.level = Level::Pumped;
                                                self.photons.swap_remove(i);
                                            }
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                i += 1;
            }

            for ((i, j), a) in self.atoms.indexed_iter_mut() {
                match a.level {
                    Level::Charged => {
                        if spon_dist.sample(&mut self.rng) {
                            if matches!(self.levels, Levels::Four) {
                                a.level = Level::Uncharged;
                            } else {
                                a.level = Level::Ground;
                            }
                            let x = i as f32;
                            let y = j as f32;
                            let pos = Vec2::new(x, y);
                            let vel =
                                PHOTON_SPEED * Vec2::from_angle(angle_dist2.sample(&mut self.rng));
                            let p = Photon {
                                pos,
                                vel,
                                ty: PhotonType::Spont,
                                escaped: false,
                            };
                            self.photons.push(p);
                        }
                    }
                    Level::Pumped => {
                        if pump_drop_dist.sample(&mut self.rng) {
                            a.level = Level::Charged;
                        }
                    }
                    Level::Uncharged => {
                        if decharge_drop_dist.sample(&mut self.rng) {
                            a.level = Level::Ground;
                        }
                    }
                    _ => {}
                }
            }
            for p in &mut self.photons {
                p.pos += p.vel * dt;

                if p.pos.x > N_ATOMS_X as f32 && !p.escaped {
                    if reflect_dist.sample(&mut self.rng) {
                        p.vel.x = -p.vel.x.abs();
                        p.vel.y += reflect_noise.sample(&mut self.rng) as f32;
                        p.vel = p.vel.normalize();
                    } else {
                        p.escaped = true;
                        if p.ty == PhotonType::Stim {
                            *self.escaped_buffer.get_mut(0).unwrap() += 1.;
                        }
                    }
                } else if p.pos.x < PHOTON_SPEED * self.dt {
                    p.vel.x = p.vel.x.abs();
                    p.vel.y += reflect_noise.sample(&mut self.rng) as f32;
                }
            }
            self.iter += 1;
        }
    }
}
