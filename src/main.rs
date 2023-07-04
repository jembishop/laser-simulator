use std::collections::VecDeque;
use std::ops::{RangeBounds, RangeInclusive};

use egui::plot::{Line, Plot, Value, Values};
use nannou::color::IntoLinSrgba;
use nannou::{color::Mix, prelude::*};
use nannou_egui::egui::{Color32, Style, Visuals};
use nannou_egui::{egui, Egui};
use ndarray::prelude::*;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::{
    distributions::{Bernoulli, Uniform},
    prelude::Distribution,
    rngs::ThreadRng,
    thread_rng,
};
// use egui::plot::{Line, Plot, PlotPoints};
use rand_distr::Poisson;

fn main() {
    nannou::app(model).update(update).run();
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

struct Model {
    _window: window::Id,
    atoms: Array2<Atom>,
    photons: Vec<Photon>,
    iter: usize,
    rng: ThreadRng,
    levels: Levels,
    egui: Egui,

    spont_chance: f32,
    reflect_chance: f32,
    interact_chance: f32,
    pump_rate: f32,
    pump_drop_chance: f32,
    decharged_drop_chance: f32,

    q: f32,
    dt: f32,
    t: f32,
    level_pop_buffer: VecDeque<[f32; 4]>,
    time_buffer: VecDeque<f32>,
    escaped_buffer: VecDeque<f32>,
}

const N_ATOMS_X: usize = 88;
const N_ATOMS_Y: usize = 20;

const DRAW_OFFSET_X: f32 = -60.;
const DRAW_OFFSET_Y: f32 = -200.;

const SCREEN_SIZE: [u32; 2] = [1080, 720];
const ATOM_CELL_SIZE: f32 = 10.;

const PHOTON_SPEED: f32 = 10.;

const PUMP_HEIGHT: f32 = 20.;
const PUMP_WIDTH: f32 = (N_ATOMS_X as f32) * ATOM_CELL_SIZE;
const PUMP_Y: f32 = (N_ATOMS_Y as f32) * ATOM_CELL_SIZE / 2. + PUMP_HEIGHT;

const MIRROR_HEIGHT: f32 = (N_ATOMS_Y as f32) * (ATOM_CELL_SIZE + 5.);
const MIRROR_WIDTH: f32 = 10.;

const ANGLE_PAD: f32 = 1.;
const LEVEL_COLORS: [Rgb<u8>; 4] = [GRAY, DARKRED, RED, GREEN];

const MIRROR_X: f32 = PUMP_WIDTH / 2. + MIRROR_WIDTH;

const REFLECT_Y_NOISE: f64 = 0.5;

const MAX_PUMP: f32 = 100.;

const BUFFER_SIZE: usize = 2048;

const AVG_WINDOW: usize = 64;

fn model(app: &App) -> Model {
    let _window = app
        .new_window()
        .size(SCREEN_SIZE[0], SCREEN_SIZE[1])
        .view(view)
        .raw_event(raw_window_event)
        .build()
        .unwrap();
    let atoms = Array2::from_elem(
        (N_ATOMS_X, N_ATOMS_Y),
        Atom {
            level: Level::Ground,
        },
    );
    let photons = Vec::new();
    let rng = thread_rng();
    let levels = Levels::Four;
    let window = app.window(_window).unwrap();

    let egui = Egui::from_window(&window);
    let level_pop_buffer = VecDeque::new();
    let time_buffer = VecDeque::new();
    let escaped_buffer = VecDeque::from_iter(std::iter::repeat(0.).take(AVG_WINDOW));

    Model {
        _window,
        atoms,
        photons,
        iter: 0,
        t: 0.,
        levels,
        rng,
        egui,
        dt: 0.3,
        q: 0.98,

        pump_rate: 10.,
        spont_chance: 0.004,
        reflect_chance: 0.5,
        interact_chance: 0.4,
        pump_drop_chance: 0.05,
        decharged_drop_chance: 0.1,
        level_pop_buffer,
        time_buffer,
        escaped_buffer,
    }
}

fn offset() -> Vec2 {
    -Vec2::new(
        (N_ATOMS_X as f32 * ATOM_CELL_SIZE) / 2.,
        (N_ATOMS_Y as f32 * ATOM_CELL_SIZE) / 2.,
    ) + ATOM_CELL_SIZE / 2.
}
fn raw_window_event(_app: &App, model: &mut Model, event: &nannou::winit::event::WindowEvent) {
    model.egui.handle_raw_event(event);
}

fn update(_app: &App, _model: &mut Model, _update: Update) {
    let dt = _model.dt;
    let new_dist = Poisson::new(f32::max(_model.pump_rate * dt, 0.01)).unwrap();
    let angle_dist = Uniform::new(-PI / 2. + ANGLE_PAD, PI / 2. - ANGLE_PAD);
    let angle_dist2 = Uniform::new(0., PI);
    let len_dist = Uniform::new(-PUMP_WIDTH / 2., PUMP_WIDTH / 2.);
    let interact_dist = Bernoulli::new((_model.interact_chance * dt) as f64).unwrap();
    let spon_dist = Bernoulli::new((_model.spont_chance as f64) * (dt as f64)).unwrap();
    let pump_drop_dist = Bernoulli::new((_model.pump_drop_chance * dt) as f64).unwrap();
    let decharge_drop_dist = Bernoulli::new((_model.decharged_drop_chance * dt) as f64).unwrap();
    let reflect_dist = Bernoulli::new(_model.reflect_chance as f64).unwrap();
    let reflect_noise = Uniform::new(-REFLECT_Y_NOISE, REFLECT_Y_NOISE);
    let q_dist = Bernoulli::new((dt * (1. - _model.q)) as f64).unwrap();

    let pop_counts = _model.atoms.iter().fold([0; 4], |mut cum, x| {
        cum[x.level as usize] += 1;
        cum
    });
    let pop_counts_f = pop_counts.map(|x| (x as f32) / (_model.atoms.len() as f32));
    let egui = &mut _model.egui;
    egui.set_elapsed_time(_update.since_start);
    let ctx = egui.begin_frame();
    // let style: Style = Style { 
    //     visuals: Visuals {
    //         ..Visuals::default()
    //     },
    //     ..Style::default()
    // };
    // ctx.set_style(style);
    egui::CentralPanel::default().show(&ctx, |ui| {
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                // Resolution slider
                ui.style_mut().spacing.slider_width = 400.0;
                ui.label("Pump Rate:");
                ui.add(egui::Slider::new(
                    &mut _model.pump_rate,
                    std::ops::RangeInclusive::new(0.1, MAX_PUMP),
                ));
                ui.label("Spont Prob:");
                ui.add(egui::Slider::new(
                    &mut _model.spont_chance,
                    std::ops::RangeInclusive::new(0., 1.),
                ));

                ui.label("Reflect Prob:");
                ui.add(egui::Slider::new(
                    &mut _model.reflect_chance,
                    std::ops::RangeInclusive::new(0., 1.),
                ));

                ui.label("dt");
                ui.add(egui::Slider::new(
                    &mut _model.dt,
                    std::ops::RangeInclusive::new(0., 1.),
                ));

                ui.label("Q");
                ui.add(egui::Slider::new(
                    &mut _model.q,
                    std::ops::RangeInclusive::new(0.5, 0.99),
                ));

                ui.label("Interact Prob:");
                ui.add(egui::Slider::new(
                    &mut _model.interact_chance,
                    std::ops::RangeInclusive::new(0., 1.),
                ));

                ui.label("Pump Drop Prob:");
                ui.add(egui::Slider::new(
                    &mut _model.pump_drop_chance,
                    std::ops::RangeInclusive::new(0., 1.),
                ));

                ui.label("Decharge Drop Prob:");
                ui.add(egui::Slider::new(
                    &mut _model.decharged_drop_chance,
                    std::ops::RangeInclusive::new(0., 1.),
                ));

                egui::ComboBox::from_label("Number of Levels")
                    .selected_text(format!("{:?} Levels", _model.levels))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut _model.levels, Levels::Two, "Two Levels");
                        ui.selectable_value(&mut _model.levels, Levels::Three, "Three Levels");
                        ui.selectable_value(&mut _model.levels, Levels::Four, "Four Levels");
                    });
            });

            ui.vertical(|ui| {
                let mut lines = (0..4).map(|i| {
                    let vs: Vec<Value> = _model
                        .level_pop_buffer
                        .iter()
                        .map(|x| x[i])
                        .zip(_model.time_buffer.iter())
                        .map(|(y, x)| Value {
                            x: *x as f64,
                            y: y as f64,
                        })
                        .collect();
                    let col = LEVEL_COLORS[i];
                    let col3: Color32 = Color32::from_rgb(col.red, col.green, col.blue);
                    let line = Line::new(Values::from_values(vs)).stroke((2., col3));
                    line
                });
                ui.add(
                    Plot::new("Pop prop")
                        .line(lines.next().unwrap())
                        .line(lines.next().unwrap())
                        .line(lines.next().unwrap())
                        .line(lines.next().unwrap())
                        .view_aspect(4.0)
                        .include_y(1.0)
                        .include_y(0.),
                );

                ui.add_space(30.);
                let escaped_mv_vec: Vec<f32> = _model.escaped_buffer.iter().copied().collect();
                let escaped_mv_avg: Vec<f32> = escaped_mv_vec
                    .windows(AVG_WINDOW + 1)
                    .map(|x| x.iter().sum::<f32>() / ((AVG_WINDOW as f32) * f32::max(_model.dt, 0.001)))
                    .collect();
                debug_assert!(escaped_mv_avg.len() == _model.time_buffer.len());
                let escaped: Vec<Value> = escaped_mv_avg
                    .iter()
                    .zip(_model.time_buffer.iter())
                    .map(|(y, x)| Value {
                        x: *x as f64,
                        y: *y as f64,
                    })
                    .collect();
                let col = WHITE;
                let col3: Color32 = Color32::from_rgb(col.red, col.green, col.blue);
                let line = Line::new(Values::from_values(escaped)).stroke((2., col3));
                ui.add(
                    Plot::new("Intensity")
                        .line(line)
                        .view_aspect(4.0)
                        .include_y(0.),
                );
                ui.set_row_height(80.);
            })
        });
    });

    if dt == 0. {
        return;
    }

    _model.level_pop_buffer.push_front(pop_counts_f);
    _model.time_buffer.push_front(_model.t);
    if _model.level_pop_buffer.len() > BUFFER_SIZE {
        _model.level_pop_buffer.pop_back();
        _model.time_buffer.pop_back();
    }

    _model.escaped_buffer.push_front(0.);
    if _model.escaped_buffer.len() > BUFFER_SIZE + AVG_WINDOW {
        _model.escaped_buffer.pop_back();
    }

    for _ in 0..4 {
        _model.t += dt;
        let new = new_dist.sample(&mut _model.rng) as usize;
        for dir in [-1., 1.] {
            for _ in 0..new / 2 {
                let angle = angle_dist.sample(&mut _model.rng);
                let len = len_dist.sample(&mut _model.rng);
                let p = Photon {
                    pos: Vec2::new(len, dir * PUMP_Y),
                    vel: -PHOTON_SPEED * Vec2::Y.rotate(angle) * dir,
                    ty: PhotonType::Pump,
                    escaped: false,
                };
                _model.photons.push(p);
            }
        }
        // let win_rect = _app.window_rect();
        let offset = offset();
        let mut i = 0;
        while i < _model.photons.len() {
            if _model.photons[i].pos.x.abs() > (SCREEN_SIZE[0]) as f32
                || _model.photons[i].pos.y.abs() > (PUMP_Y) as f32
            {
                _model.photons.swap_remove(i);
                i += 1;
                continue;
            }
            let v = (_model.photons[i].pos - offset) / ATOM_CELL_SIZE;
            if v.x < N_ATOMS_X as f32 && v.y < N_ATOMS_Y as f32 && v.x > 0. && v.y > 0. {
                let idx = (v.x as usize, v.y as usize);
                let a = &mut _model.atoms[idx];
                if q_dist.sample(&mut _model.rng) {
                    _model.photons.swap_remove(i);
                    i += 1;
                    continue;
                }
                if interact_dist.sample(&mut _model.rng) {
                    {
                        let p = &_model.photons[i];
                        match a.level {
                            Level::Charged => {
                                let emit = match _model.levels {
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
                                                * Vec2::Y
                                                    .rotate(angle_dist2.sample(&mut _model.rng)),
                                        vel: p.vel,
                                        ty: PhotonType::Stim,
                                        escaped: false,
                                    };
                                    _model.photons.push(p);
                                }
                            }
                            Level::Uncharged => {
                                if matches!(_model.levels, Levels::Four) {
                                    _model.photons.swap_remove(i);
                                    a.level = Level::Charged;
                                }
                            }
                            Level::Ground => {
                                // level 2 always charge
                                // level 3 blue pump, normal charge
                                // level 4 only blue pump
                                match _model.levels {
                                    Levels::Two => {
                                        a.level = Level::Charged;
                                        _model.photons.swap_remove(i);
                                    }
                                    Levels::Three => {
                                        if _model.photons[i].ty == PhotonType::Pump {
                                            a.level = Level::Pumped;
                                        } else {
                                            a.level = Level::Charged;
                                        }
                                        _model.photons.swap_remove(i);
                                    }
                                    Levels::Four => {
                                        if _model.photons[i].ty == PhotonType::Pump {
                                            a.level = Level::Pumped;
                                            _model.photons.swap_remove(i);
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

        for ((i, j), a) in _model.atoms.indexed_iter_mut() {
            match a.level {
                Level::Charged => {
                    if spon_dist.sample(&mut _model.rng) {
                        if matches!(_model.levels, Levels::Four) {
                            a.level = Level::Uncharged;
                        } else {
                            a.level = Level::Ground;
                        }
                        let x = i as f32;
                        let y = j as f32;
                        let pos = Vec2::new(x, y) * ATOM_CELL_SIZE + offset;
                        let vel =
                            PHOTON_SPEED * Vec2::Y.rotate(angle_dist2.sample(&mut _model.rng));
                        let p = Photon {
                            pos,
                            vel,
                            ty: PhotonType::Spont,
                            escaped: false,
                        };
                        _model.photons.push(p);
                    }
                }
                Level::Pumped => {
                    if pump_drop_dist.sample(&mut _model.rng) {
                        a.level = Level::Charged;
                    }
                }
                Level::Uncharged => {
                    if decharge_drop_dist.sample(&mut _model.rng) {
                        a.level = Level::Ground;
                    }
                }
                _ => {}
            }
        }
        for p in &mut _model.photons {
            p.pos += p.vel * dt;

            if p.pos.x > MIRROR_X && !p.escaped {
                if reflect_dist.sample(&mut _model.rng) {
                    p.vel.x = -p.vel.x.abs();
                    p.vel.y += reflect_noise.sample(&mut _model.rng) as f32;
                } else {
                    p.escaped = true;
                    *_model.escaped_buffer.get_mut(0).unwrap() += 1.;
                }
            } else if p.pos.x < -MIRROR_X + PHOTON_SPEED * _model.dt {
                p.vel.x = p.vel.x.abs();
                p.vel.y += reflect_noise.sample(&mut _model.rng) as f32;
            }
        }
        _model.iter += 1;
    }
}

fn view(app: &App, _model: &Model, frame: Frame) {
    // app.main_window()
    //     .capture_frame(format!("frames/{:04}.png", _model.iter));
    let draw = app.draw();
    // draw.background().color(BLACK);
    // let win_d = win_rect.top_right() - win_rect.bottom_left();
    let offset = offset();

    let draw_off = Vec2::new(DRAW_OFFSET_X, DRAW_OFFSET_Y);

    for ((i, j), a) in _model.atoms.indexed_iter() {
        let x = i as f32;
        let y = j as f32;
        let v = Vec2::new(x, y) * ATOM_CELL_SIZE + offset;
        let col = LEVEL_COLORS[a.level as usize];
        draw.ellipse()
            .xy(v + draw_off)
            .wh(Vec2::ONE * (ATOM_CELL_SIZE - 1.))
            .color(col);
    }

    for p in &_model.photons {
        let col = match p.ty {
            PhotonType::Pump => WHITE,
            PhotonType::Stim => ORANGE,
            PhotonType::Spont => YELLOW,
        };

        draw.ellipse()
            .xy(p.pos + draw_off)
            .wh(Vec2::ONE * (ATOM_CELL_SIZE / 2.))
            .color(col);
    }

    let pump_col = STEELBLUE;
    draw.rect()
        .w_h(PUMP_WIDTH, PUMP_HEIGHT)
        .xy(Vec2::new(0., PUMP_Y) + draw_off)
        .color(pump_col);
    draw.rect()
        .w_h(PUMP_WIDTH, PUMP_HEIGHT)
        .xy(Vec2::new(0., -PUMP_Y) + draw_off)
        .color(pump_col);

    let mirror_col: LinSrgb = WHITE.into_lin_srgba().into();
    let g: LinSrgb = STEELBLUE.into_lin_srgba().into();
    draw.rect()
        .w_h(MIRROR_WIDTH, MIRROR_HEIGHT)
        .xy(Vec2::new(MIRROR_X, 0.) + draw_off)
        .color(mirror_col.mix(&g, 1. - _model.reflect_chance));

    draw.rect()
        .w_h(MIRROR_WIDTH, MIRROR_HEIGHT)
        .xy(Vec2::new(-MIRROR_X, 0.) + draw_off)
        .color(mirror_col);

    _model.egui.draw_to_frame(&frame).unwrap();
    draw.to_frame(app, &frame).unwrap();
}
