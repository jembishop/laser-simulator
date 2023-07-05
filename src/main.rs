use std::collections::VecDeque;
use std::f32::consts::PI;

use egui::plot::{Line, Plot, PlotPoints};
use egui::{Color32, Pos2, Rect, Stroke};

use glam::Vec2;
use ndarray::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::{
    distributions::{Bernoulli, Uniform},
    prelude::Distribution,
    rngs::ThreadRng,
    thread_rng,
};
use rand_distr::Poisson;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(SCREEN_SIZE[0] as f32, SCREEN_SIZE[1] as f32)),
        ..Default::default()
    };
    eframe::run_native(
        "Laser Simulator",
        options,
        Box::new(|_cc| Box::<App>::default()),
    )
}

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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

struct App {
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
}

const N_ATOMS_X: usize = 88;
const N_ATOMS_Y: usize = 24;

const DRAW_OFFSET_X: f32 = 480.;
const DRAW_OFFSET_Y: f32 = 520.;

const SCREEN_SIZE: [u32; 2] = [1080, 720];
const ATOM_CELL_SIZE: f32 = 10.;

const PHOTON_SPEED: f32 = 10.;

const PUMP_HEIGHT: f32 = 20.;
const PUMP_WIDTH: f32 = (N_ATOMS_X as f32) * ATOM_CELL_SIZE;
const PUMP_Y: f32 = (N_ATOMS_Y as f32) * ATOM_CELL_SIZE / 2. + PUMP_HEIGHT / 2.;

const MIRROR_HEIGHT: f32 = (N_ATOMS_Y as f32) * (ATOM_CELL_SIZE + 1.);
const MIRROR_WIDTH: f32 = 10.;

const ANGLE_PAD: f32 = 1.;
const LEVEL_COLORS: [Color32; 4] = [
    Color32::GRAY,
    Color32::DARK_RED,
    Color32::RED,
    Color32::GOLD,
];

const MIRROR_X: f32 = PUMP_WIDTH / 2. + MIRROR_WIDTH;

const REFLECT_Y_NOISE: f64 = 0.5;

const MAX_PUMP: f32 = 200.;

const BUFFER_SIZE: usize = 2048;

const AVG_WINDOW: usize = 256;

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
        let levels = Levels::Four;

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
            dt: 0.3,
            q: 0.98,
            show_pump_photons: true,
            pump_rate: 20.,
            spont_chance: 0.01,
            reflect_chance: 0.5,
            interact_chance: 0.05,
            pump_drop_chance: 0.3,
            decharged_drop_chance: 0.4,

            level_pop_buffer,
            time_buffer,
            escaped_buffer,
        }
    }
}

fn offset() -> Vec2 {
    -Vec2::new(
        (N_ATOMS_X as f32 * ATOM_CELL_SIZE) / 2.,
        (N_ATOMS_Y as f32 * ATOM_CELL_SIZE) / 2.,
    ) + ATOM_CELL_SIZE / 2.
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        let dt = self.dt;
        let new_dist = Poisson::new(f32::max(self.pump_rate * dt, 0.01)).unwrap();
        let angle_dist = Uniform::new(-PI / 2. + ANGLE_PAD, PI / 2. - ANGLE_PAD);
        let angle_dist2 = Uniform::new(0., PI);
        let len_dist = Uniform::new(-PUMP_WIDTH / 2., PUMP_WIDTH / 2.);
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
                        .view_aspect(4.0)
                        .link_axis("g1", true, false)
                        .include_y(1.0)
                        .include_y(0.)
                        .show(ui, |plot_ui| {
                            for l in lines {
                                plot_ui.line(l)
                            }
                        });

                    ui.label("Avg Power");
                    let escaped_mv_vec: Vec<f32> = self.escaped_buffer.iter().copied().collect();
                    let escaped_mv_avg: Vec<f32> = escaped_mv_vec
                        .windows(AVG_WINDOW + 1)
                        .map(|x| {
                            x.iter().sum::<f32>() / ((AVG_WINDOW as f32) * f32::max(self.dt, 0.001))
                        })
                        .collect();

                    debug_assert!(escaped_mv_avg.len() == self.time_buffer.len());

                    let escaped: PlotPoints = escaped_mv_avg
                        .iter()
                        .zip(self.time_buffer.iter())
                        .map(|(y, x)| [*x as f64, *y as f64])
                        .collect();
                    let col = Color32::WHITE;
                    let line = Line::new(escaped).stroke((2., col));
                    Plot::new("Intensity")
                        .view_aspect(4.0)
                        .link_axis("g1", true, false)
                        .include_y(0.)
                        .show(ui, |plot_ui| {
                            plot_ui.line(line);
                        });

                    ui.set_row_height(80.);
                })
            });
            let painter = ui.painter();
            let offset = offset();
            let draw_off = Vec2::new(DRAW_OFFSET_X, DRAW_OFFSET_Y);
            for ((i, j), a) in self.atoms.indexed_iter() {
                let x = i as f32;
                let y = j as f32;
                let v = Vec2::new(x, y) * ATOM_CELL_SIZE + offset;
                let col = LEVEL_COLORS[a.level as usize];
                let draw_v = v + draw_off;
                let loc = Pos2::new(draw_v.x, draw_v.y);
                painter.circle(
                    loc,
                    ATOM_CELL_SIZE / 2.,
                    col,
                    Stroke::new(0., Color32::BLACK),
                );
            }
            for p in &self.photons {
                let col = match p.ty {
                    PhotonType::Pump => Color32::WHITE,
                    PhotonType::Stim => Color32::GREEN,
                    PhotonType::Spont => Color32::YELLOW,
                };

                let draw_v = p.pos + draw_off;
                let loc = Pos2::new(draw_v.x, draw_v.y);
                if !(p.ty == PhotonType::Pump && !self.show_pump_photons) {
                    painter.circle(
                        loc,
                        ATOM_CELL_SIZE / 4.,
                        col,
                        Stroke::new(0., Color32::BLACK),
                    );
                }
            }
            for left in [true, false] {
                let (x, col) = if left {
                    (-MIRROR_X, Color32::WHITE)
                } else {
                    (
                        MIRROR_X,
                        Color32::WHITE.linear_multiply(self.reflect_chance + 0.1),
                    )
                };
                let min = Vec2::new(x - MIRROR_WIDTH / 2., -MIRROR_HEIGHT / 2.);
                let max = Vec2::new(x + MIRROR_WIDTH / 2., MIRROR_HEIGHT / 2.);
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
            for dir in [-1., 1.] {
                for _ in 0..new / 2 {
                    let angle = angle_dist.sample(&mut self.rng);
                    let len = len_dist.sample(&mut self.rng);
                    let p = Photon {
                        pos: Vec2::new(len, dir * PUMP_Y),
                        vel: -PHOTON_SPEED * Vec2::Y.rotate(Vec2::from_angle(angle)) * dir,
                        ty: PhotonType::Pump,
                        escaped: false,
                    };
                    self.photons.push(p);
                }
            }
            let offset = offset();
            let mut i = 0;
            while i < self.photons.len() {
                if self.photons[i].pos.x.abs() > (SCREEN_SIZE[0]) as f32
                    || self.photons[i].pos.y.abs() > (PUMP_Y) as f32
                {
                    self.photons.swap_remove(i);
                    i += 1;
                    continue;
                }
                let v = (self.photons[i].pos - offset) / ATOM_CELL_SIZE;
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
                                                + 2. * ATOM_CELL_SIZE
                                                    * Vec2::Y.rotate(Vec2::from_angle(
                                                        angle_dist2.sample(&mut self.rng),
                                                    )),
                                            vel: p.vel,
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
                            let pos = Vec2::new(x, y) * ATOM_CELL_SIZE + offset;
                            let vel = PHOTON_SPEED
                                * Vec2::Y
                                    .rotate(Vec2::from_angle(angle_dist2.sample(&mut self.rng)));
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

                if p.pos.x > MIRROR_X && !p.escaped {
                    if reflect_dist.sample(&mut self.rng) {
                        p.vel.x = -p.vel.x.abs();
                        p.vel.y += reflect_noise.sample(&mut self.rng) as f32;
                    } else {
                        p.escaped = true;
                        if p.ty == PhotonType::Stim {
                            *self.escaped_buffer.get_mut(0).unwrap() += 1.;
                        }
                    }
                } else if p.pos.x < -MIRROR_X + PHOTON_SPEED * self.dt {
                    p.vel.x = p.vel.x.abs();
                    p.vel.y += reflect_noise.sample(&mut self.rng) as f32;
                }
            }
            self.iter += 1;
        }
    }
}
