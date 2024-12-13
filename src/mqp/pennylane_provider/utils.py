int2bit = lambda x, N: str(bin(x)[2:].zfill(N))
bit2int = lambda b: int("".join(str(bs) for bs in b), base=2)
