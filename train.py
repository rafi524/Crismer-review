import argparse
import pandas as pd
import torch

from trainer import trainer
from utills import eval
from utills import one_hot_features   # adjust import if needed


def main():

    parser = argparse.ArgumentParser(description="Train CRISPR Transformer Model")

    parser.add_argument(
        "--train_data",
        type=str,
        required=True,
        help="Path to changeseq_siteseq.csv (training dataset)"
    )

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to testing dataset (e.g., circleseq_all.csv)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="aot_idt_weights_three.pth",
        help="Path to save trained model weights"
    )

    args = parser.parse_args()

    print("Loading training dataset...")
    train_df = pd.read_csv(args.train_data)

    print("Preparing training features...")
    train_x = one_hot_features(train_df)
    train_y = train_df['Active'].to_numpy()

    config = {
        'num_layers': 2,
        'num_heads': 4,
        'number_hidder_layers': 2,
        'dropout_prob': 0.2,
        'batch_size': 128,
        'epochs': 50,
        'learning_rate': 0.001,
        'pos_weight': 30,
        'attn': False
    }

    print("Starting training...")
    model, history = trainer(config, train_x, train_y)

    print("Training finished")

    print("Loading test dataset...")
    test_df = pd.read_csv(args.test_data)

    print("Running evaluation...")
    eval(model, test_df)

    print("Saving model weights...")
    torch.save(model.state_dict(), args.output)

    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()