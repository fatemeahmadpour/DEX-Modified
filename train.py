import hydra
from dex.trainers.rl_trainer import RLTrainer


@hydra.main(version_base=None, config_path="./dex/configs", config_name="train")
def main(cfg):
    exp = RLTrainer(cfg)
    exp.train()

if __name__ == "__main__":
    main()