from models.decom import RICD, ReconstructionDecoder


def count_params(module):
    return sum(parameter.numel() for parameter in module.parameters())


ricd = RICD(channels=64)
decoder = ReconstructionDecoder(channels=64)

ricd_params = count_params(ricd)
decoder_params = count_params(decoder)

print(f"RICD: {ricd_params:,} ({ricd_params / 1e6:.6f} M)")
print(f"Decoder: {decoder_params:,} ({decoder_params / 1e6:.6f} M)")
print(f"Combined: {ricd_params + decoder_params:,}")

assert ricd_params == 2_984_327
assert decoder_params == 4_084_995
assert ricd_params + decoder_params == 7_069_322

