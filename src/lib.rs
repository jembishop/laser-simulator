mod laser;

use std::collections::VecDeque;
use std::f32::consts::PI;

use egui::plot::{Line, Plot, PlotPoints};
use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Stroke};

use glam::Vec2;
use rand::Rng;
use rand::{distributions::Uniform, prelude::Distribution, rngs::ThreadRng, thread_rng};

use laser::{Laser, Levels, PhotonType, N_ATOMS_PAD, N_ATOMS_X, N_ATOMS_Y};

const MIRROR_WIDTH: f32 = 10.;
const VIEW_ASPECT: f32 = 6.;

const LEVEL_COLORS: [Color32; 4] = [
    Color32::DARK_GRAY,
    Color32::DARK_RED,
    Color32::RED,
    Color32::GOLD,
];
const PHOTON_COLORS: [Color32; 3] = [Color32::WHITE, Color32::GREEN, Color32::YELLOW];

const DT_ANIM: f32 = 0.008;

const MAX_PUMP: f32 = 400.;

const BUFFER_SIZE: usize = 2048;

const AVG_WINDOW: usize = 256;

pub struct App {
    rng: ThreadRng,
    laser: Laser,
    dt: f32,
    t: f32,
    show_pump_photons: bool,
    level_pop_buffer: VecDeque<[f32; 4]>,
    time_buffer: VecDeque<f32>,
    escaped_buffer: VecDeque<f32>,
    show_graph: bool,
    diagram_rand_dir: Vec2,
    stim_col: usize,
    t_anim: f32,
}

fn two_level_text(ui: &mut egui::Ui, t: f32, rand_dir: Vec2, stim_col: usize) {
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
                 In the two level laser the pump and emitted photons are the same energy, though in the simulation we colour the photons according to their source: ("
                );
        ui.label(RichText::new("pump,").color(PHOTON_COLORS[0]));
        ui.label(RichText::new("spontaneous,").color(PHOTON_COLORS[1]));
        ui.label(RichText::new("stimulated").color(PHOTON_COLORS[2]));
        ui.label(").You will notice the laser output doesn't look very good at the moment. In fact it's impossible to get the two level laser to work! \
                 try putting the laser in the 3 level state using the dropdown on the left, to see more info."
        );
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
            "To see why we need the three level laser we need to understand  what makes a laser work in the first place. \
            A laser requires a higher proportion of the atoms in the  charged state than the ground state. \
            Once this is condition is achieved (known as population inversion) \
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
            The laser should be 'lasing' now, when your ready try the four level to see how a real laser works!"
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
    painter.circle(
        offset3 + perp,
        rad2,
        PHOTON_COLORS[0],
        Stroke::new(0., col1),
    );
}

fn four_level_text(ui: &mut egui::Ui) {
    ui.horizontal_wrapped(|ui| {
        ui.label(
            "The three level laser works, but you won't see many lasers using this mechanism in practice. Why? \
             The problem is that the new photons we produce from stimulated emission can still be absorbed by the \
             ground state atoms, which then can waste the energy in spontaneous emission. To fix this we would ideally \
             like the stimulated photons to not interfere with the ground state atoms, similar to how we don't want the \
             pumped photons to interfere with the charged atoms. We can do this by making the charged state transition \
             not to the ground state but a slightly above ground state. This slightly higher state should then quickly \
             decay to the ground state so it can be pumped back up again. I call this state the"
        );
        ui.label(RichText::new("decharged").color(LEVEL_COLORS[1]));
        ui.label(
            "state. Now the stimulated photons can travel through the medium wihout risk of being \
            reabsorbed by ground state atoms, as they have the wrong energy for the ground-charged transition. \
            In normal lasers the maximum pump rate is much lower than the maximum in this simulation, so the four level laser \
            is the only practical design.\n You can see the graphs by clicking the next tab, which show the relative proportion of atoms \
            in their respective states, and the smoothed power output ie the number of photons which make it out \
            of the laser exit mirror. In the panel to the left you can see controls to fiddle around with \
            the parameters of the laser. The value of Q is the 'quality' of the medium, a lower Q will mean \
            more photons are absorbed naturally in the material. You will notice that lowering Q can increase \
            the proportion of atoms in the charged state, by dampening the stimulated emission buildup. Raising \
            Q rapidly can give you a big pulse as there is a bigger population inversion intially. This phenomenon is \
            known as Q switching. Note also how the laser output varies as you alter the exit mirror reflectivity, \
            more reflectivity allows more chances to stimulate charged atoms, but also more photons confined within the laser \
            and a balance must be struck. At high pump levels you don't even need the exit mirror, such lasers are called \
            superluminescent.
        ");
    });
}

impl Default for App {
    fn default() -> Self {
        let rng = thread_rng();

        let level_pop_buffer = VecDeque::new();
        let time_buffer = VecDeque::new();
        let escaped_buffer = VecDeque::from_iter(std::iter::repeat(0.).take(AVG_WINDOW));
        let laser = Laser::new();
        Self {
            laser,
            rng,
            t: 0.,
            dt: 0.2,
            show_pump_photons: true,
            level_pop_buffer,
            time_buffer,
            escaped_buffer,

            show_graph: false,
            diagram_rand_dir: Vec2::ONE,
            t_anim: 0.,
            stim_col: 0,
        }
    }
}

impl App {
    fn draw_controls(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.style_mut().spacing.slider_width = 400.0;
            ui.label("Pump Rate:");
            ui.add(egui::Slider::new(
                &mut self.laser.pump_rate,
                std::ops::RangeInclusive::new(0.1, MAX_PUMP),
            ));
            ui.label("Spontaneous Emission Probability:");
            ui.add(egui::Slider::new(
                &mut self.laser.spont_chance,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.label("Mirror Reflection Probability:");
            ui.add(egui::Slider::new(
                &mut self.laser.reflect_chance,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.label("dt");
            ui.add(egui::Slider::new(
                &mut self.dt,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.label("Q");
            ui.add(egui::Slider::new(
                &mut self.laser.q,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.label("Photon Interaction Probability:");
            ui.add(egui::Slider::new(
                &mut self.laser.interact_chance,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.label("Pumped -> Charged Probability:");
            ui.add(egui::Slider::new(
                &mut self.laser.pump_drop_chance,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.label("Decharged -> Ground Probability:");
            ui.add(egui::Slider::new(
                &mut self.laser.decharged_drop_chance,
                std::ops::RangeInclusive::new(0., 1.),
            ));

            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_pump_photons, "Show Pump Photons");
                egui::ComboBox::from_label("")
                    .selected_text(format!("{:?} Levels", self.laser.levels))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.laser.levels, Levels::Two, "Two Level");
                        ui.selectable_value(&mut self.laser.levels, Levels::Three, "Three Level");
                        ui.selectable_value(&mut self.laser.levels, Levels::Four, "Four Level");
                    });
            });
        });
    }

    fn draw_graphs(&mut self, ui: &mut egui::Ui) {
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
        let escaped_mv_vec: Vec<f32> = self.escaped_buffer.iter().copied().collect();
        let escaped_mv_avg: Vec<f32> = escaped_mv_vec
            .windows(AVG_WINDOW + 1)
            .map(|x| x.iter().sum::<f32>() / ((AVG_WINDOW as f32) * f32::max(self.dt, 0.001)))
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
    }
    fn draw_info(&mut self, ui: &mut egui::Ui) {
        let uniform_angle_dist = Uniform::new(0., 2. * PI);
        if self.t_anim + DT_ANIM > 1. {
            self.t_anim = 0.;
            self.diagram_rand_dir = Vec2::from_angle(uniform_angle_dist.sample(&mut self.rng));
            self.stim_col = self.rng.gen::<usize>() % 3;
        } else {
            self.t_anim += DT_ANIM;
        }
        match self.laser.levels {
            Levels::Two => {
                two_level_text(ui, self.t_anim, self.diagram_rand_dir, self.stim_col);
            }
            Levels::Three => {
                three_level_text(ui, self.t_anim, self.diagram_rand_dir);
            }
            Levels::Four => {
                four_level_text(ui);
            }
        }
    }
    fn draw_laser(&mut self, ui: &mut egui::Ui) {
        // Atoms
        let (r, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::hover());
        let laser_zone = Vec2::new(r.rect.width(), r.rect.height());
        let atom_size = f32::min(
            laser_zone.x / ((N_ATOMS_X + N_ATOMS_PAD) as f32),
            laser_zone.y / ((N_ATOMS_Y + N_ATOMS_PAD) as f32),
        );
        let rt = r.rect.left_top() + egui::Vec2::new(10., 0.);
        let pad = Vec2::new(0., 0.1 * laser_zone.y);
        let draw_off = Vec2::new(rt.x, rt.y) + pad;
        for ((i, j), a) in self.laser.atoms.indexed_iter() {
            let x = i as f32;
            let y = j as f32;
            let v = Vec2::new(x, y) * atom_size;
            let col = LEVEL_COLORS[a.level as usize];
            let draw_v = v + draw_off + atom_size / 2.;
            let loc = Pos2::new(draw_v.x, draw_v.y);
            painter.circle(loc, atom_size / 2., col, Stroke::new(0., Color32::BLACK));
        }

        // Photons
        for p in &self.laser.photons {
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
                    Color32::WHITE.linear_multiply(self.laser.reflect_chance + 0.1),
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
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();
        let dt = self.dt;
        egui::CentralPanel::default().show(&ctx, |ui| {
            ui.visuals_mut().override_text_color = Some(Color32::WHITE.linear_multiply(0.5));
            // ui.visuals_mut().f
            ui.horizontal(|ui| {
                self.draw_controls(ui);
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
                        self.draw_graphs(ui);
                    } else {
                        self.draw_info(ui);
                    }
                });
            });
            self.draw_laser(ui);
        });

        if dt == 0. {
            return;
        }

        self.t += self.dt;
        let pop_counts = self.laser.atoms.iter().fold([0; 4], |mut cum, x| {
            cum[x.level as usize] += 1;
            cum
        });
        let pop_counts_f = pop_counts.map(|x| (x as f32) / (self.laser.atoms.len() as f32));
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
            let escaped = self.laser.step(self.dt, &mut self.rng);
            *self.escaped_buffer.get_mut(0).unwrap() += escaped as f32;
        }
    }
}
