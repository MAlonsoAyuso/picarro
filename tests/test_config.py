from pathlib import Path
import picarro.config
import shutil


config_example_src = Path(__file__).absolute().parent / "config_example.toml"
assert config_example_src.exists()


def test_create_config(tmp_path: Path):
    config_path = tmp_path / "config.toml"
    shutil.copyfile(config_example_src, config_path)

    conf = picarro.config.AppConfig.from_toml(config_path)
    expected_conf = picarro.config.AppConfig(
        src_dir=config_path.parent,
        results_subdir=config_path.stem,
        user=picarro.config.UserConfig(
            picarro.config.ReadConfig(
                src="../example_data/adjacent_files/**/*.dat",
                columns=["N2O", "CH4"],
                max_gap=5,
                min_length=1080,
                max_length=None,
            ),
            picarro.config.FitConfig(
                method="linear",
                t0=480,
                t0_margin=120,
                A=0.25,
                Q=4.16e-6,
                V=50e-3,
            ),
            picarro.config.OutputConfig(),
        ),
    )

    assert conf == expected_conf
