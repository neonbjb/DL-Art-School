from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter("../experiments/train_div2k_feat_rg2_more_stuff")
    f = open("../experiments/train_div2k_feat_rg2_more_stuff/console_output")
    console = f.readlines()
    search_terms = [
        ("iter", ", iter:  ", ", lr:"),
        ("l_g_total", " l_g_total: ", " switch_temperature:")
    ]
    iter = 0
    for line in console:
        if " - INFO: [epoch:" not in line:
            continue
        for name, start, end in search_terms:
            val = line[line.find(start)+len(start):line.find(end)].replace(",", "")
            if name == "iter":
                iter = int(val)
            else:
                writer.add_scalar(name, float(val), iter)
    writer.close()