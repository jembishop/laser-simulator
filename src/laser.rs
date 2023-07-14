use std::f32::consts::PI;

use glam::Vec2;
use ndarray::Array2;
use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::{
    distributions::{Bernoulli, Uniform},
    rngs::ThreadRng,
};
use rand_distr::{Distribution, Poisson};

pub const N_ATOMS_X: usize = 88;
pub const N_ATOMS_Y: usize = 24;
pub const N_ATOMS_PAD: usize = 12;
pub const PHOTON_SPEED: f32 = 1.;
pub const ANGLE_PAD: f32 = 1.;
pub const REFLECT_Y_NOISE: f64 = 0.5;

#[derive(Copy, Clone)]
pub enum Level {
    Ground,
    Uncharged,
    Charged,
    Pumped,
}

#[derive(Copy, Clone, Debug, TryFromPrimitive, PartialEq, Eq, IntoPrimitive)]
#[repr(u8)]
pub enum Levels {
    Two,
    Three,
    Four,
}

#[derive(Copy, Clone)]
pub struct Atom {
    pub level: Level,
}

#[derive(Copy, Clone, Debug, TryFromPrimitive, PartialEq, Eq, IntoPrimitive)]
#[repr(u8)]
pub enum PhotonType {
    Pump,
    Stim,
    Spont,
}

pub struct Photon {
    pub pos: Vec2,
    pub vel: Vec2,
    pub ty: PhotonType,
    pub escaped: bool,
}
pub struct Laser {
    pub atoms: Array2<Atom>,
    pub photons: Vec<Photon>,

    pub levels: Levels,
    pub spont_chance: f32,
    pub reflect_chance: f32,
    pub interact_chance: f32,
    pub pump_rate: f32,
    pub pump_drop_chance: f32,
    pub decharged_drop_chance: f32,
    pub q: f32,
}

impl Laser {
    pub fn new() -> Self {
        let atoms = Array2::from_elem(
            (N_ATOMS_X, N_ATOMS_Y),
            Atom {
                level: Level::Ground,
            },
        );
        let photons = Vec::new();
        Self {
            atoms,
            photons,
            levels: Levels::Two,
            q: 0.98,
            pump_rate: 20.,
            spont_chance: 0.02,
            reflect_chance: 0.8,
            interact_chance: 0.1,
            pump_drop_chance: 0.3,
            decharged_drop_chance: 0.4,
        }
    }
    pub fn step(&mut self, dt: f32, rng: &mut ThreadRng) -> usize {
        let uniform_angle_dist: Uniform<f32> = Uniform::new(0., 2. * PI);
        let new_dist = Poisson::new(f32::max(self.pump_rate * dt, 0.01)).unwrap();
        let angle_dist = Uniform::new(ANGLE_PAD, PI - ANGLE_PAD);
        let len_dist = Uniform::new(0., N_ATOMS_X as f32);
        let interact_dist = Bernoulli::new((self.interact_chance * dt) as f64).unwrap();
        let spon_dist = Bernoulli::new((self.spont_chance as f64) * (dt as f64)).unwrap();
        let pump_drop_dist = Bernoulli::new((self.pump_drop_chance * dt) as f64).unwrap();
        let decharge_drop_dist = Bernoulli::new((self.decharged_drop_chance * dt) as f64).unwrap();
        let reflect_dist = Bernoulli::new(self.reflect_chance as f64).unwrap();
        let reflect_noise = Uniform::new(-REFLECT_Y_NOISE, REFLECT_Y_NOISE);
        let q_dist = Bernoulli::new((dt * (1. - self.q) / 10.) as f64).unwrap();
        // self.t += dt;
        let new = new_dist.sample(rng) as usize;
        for up in [true, false] {
            for _ in 0..new / 2 {
                let (y, dir) = if up {
                    (N_ATOMS_Y as f32, 1.)
                } else {
                    (0., -1.)
                };
                let angle = angle_dist.sample(rng);
                let len = len_dist.sample(rng);
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
                if q_dist.sample(rng) {
                    self.photons.swap_remove(i);
                    i += 1;
                    continue;
                }
                if interact_dist.sample(rng) {
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
                                            + 2. * Vec2::from_angle(uniform_angle_dist.sample(rng)),
                                        vel: p.vel.normalize(),
                                        ty: PhotonType::Stim,
                                        escaped: false,
                                    };
                                    self.photons.push(p);
                                }
                            }
                            Level::Uncharged => {
                                if self.photons[i].ty != PhotonType::Pump {
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
                            Level::Pumped => {}
                        }
                    }
                }
            }
            i += 1;
        }

        for ((i, j), a) in self.atoms.indexed_iter_mut() {
            match a.level {
                Level::Charged => {
                    if spon_dist.sample(rng) {
                        if matches!(self.levels, Levels::Four) {
                            a.level = Level::Uncharged;
                        } else {
                            a.level = Level::Ground;
                        }
                        let x = i as f32;
                        let y = j as f32;
                        let pos = Vec2::new(x, y);
                        let vel = PHOTON_SPEED * Vec2::from_angle(uniform_angle_dist.sample(rng));
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
                    if pump_drop_dist.sample(rng) {
                        a.level = Level::Charged;
                    }
                }
                Level::Uncharged => {
                    if decharge_drop_dist.sample(rng) {
                        a.level = Level::Ground;
                    }
                }
                _ => {}
            }
        }
        let mut escaped = 0;
        for p in &mut self.photons {
            p.pos += p.vel * dt;

            if p.pos.x > N_ATOMS_X as f32 && !p.escaped {
                if reflect_dist.sample(rng) {
                    p.vel.x = -p.vel.x.abs();
                    p.vel.y += reflect_noise.sample(rng) as f32;
                    p.vel = p.vel.normalize();
                } else {
                    p.escaped = true;
                    if p.ty == PhotonType::Stim {
                        escaped += 1;
                    }
                }
            } else if p.pos.x < PHOTON_SPEED * dt {
                p.vel.x = p.vel.x.abs();
                p.vel.y += reflect_noise.sample(rng) as f32;
            }
        }
        escaped
    }
}
