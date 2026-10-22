"""Minimal local package workflow; pass a long-format CSV path on the command line."""

import argparse

from neural_ode import (
    create_model,
    plot_input_trajectories,
    plot_predicted_trajectories,
    plot_training_loss,
    predict,
    prep_batches,
    prep_data,
    train_fit_gmm,
    train_finetune,
    train_pretrain,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("csv_path", help="long-format trajectory CSV")
parser.add_argument("--epochs", type=int, default=20, help="epochs per training phase")
args = parser.parse_args()

data = prep_data(args.csv_path, id_col="id", time_col="time", label_col="label",
                 cutoff=0.7, cutoff_mode="fraction")
loaders = prep_batches(data, batch_size=16, val_split=0.2, normalize=True)
model = create_model(len(data.feature_names), {"latent_dim": 6, "k": 3})
pretrained = train_pretrain(loaders, model_config=model, epochs=args.epochs)
gmm = train_fit_gmm(pretrained.model, loaders)
finetuned = train_finetune(gmm.model, loaders, epochs=args.epochs)
predictions = predict(finetuned.model, loaders)
plot_input_trajectories(data)
plot_predicted_trajectories(predictions)
plot_training_loss(finetuned)
