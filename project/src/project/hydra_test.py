import hydra

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    print(cfg)

if __name__ == "__main__":
    main()
