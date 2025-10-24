from data.eeg import bcic_psd

print("1: calling bcic_psd", flush=True)
train_data, test_data, meta = bcic_psd(batch_size=1)
print("2: got loaders", flush=True)

print("3: iterating one train batch", flush=True)
batch = next(train_data)
print("4: got batch", batch["image"].shape, flush=True)