import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0.
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return 2 * loss


def discriminator_loss(disc_real_outputs, disc_fake_outputs):
    loss = 0.
    for dr, dg in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += r_loss + g_loss
    return loss


def generator_loss(disc_outputs):
    loss = 0.
    for dg in disc_outputs:
        loss += torch.mean((1 - dg) ** 2)
    return loss
