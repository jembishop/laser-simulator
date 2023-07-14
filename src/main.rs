use laser_simulator::App;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init();

    const SCREEN_SIZE: [u32; 2] = [1080, 720];
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(SCREEN_SIZE[0] as f32, SCREEN_SIZE[1] as f32)),
        ..Default::default()
    };
    eframe::run_native(
        "Laser Simulator",
        options,
        Box::new(|_| Box::new(App::default())),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                "laser_simulator",
                options,
                Box::new(|_| Box::new(App::default())),
            )
            .await
            .expect("failed to start eframe");
    });
}
