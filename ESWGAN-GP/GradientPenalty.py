from utils import * 

def gradient_penalty(discriminator, real, fake, device):
    # adds penalty to loss function to make sure input does not change too drastically
    # penalizes gradients with large norm values
    
      batch_size, c, h, w = real.shape
      alpha = torch.randn((batch_size, 1, 1, 1)).repeat(1, c, h, w).to(device)
      interplorated_images = real * alpha + fake.detach() * (1 - alpha)
      interplorated_images.requires_grad_(True)

      mixed_scores = discriminator(interplorated_images)

      gradient = torch.autograd.grad(
          inputs = interplorated_images,
          outputs = mixed_scores,
          grad_outputs = torch.ones_like(mixed_scores),
          create_graph = True,
          retain_graph = True,
      )[0]

      gradient = gradient.view(gradient.shape[0], -1)
      gradient_norm = gradient.norm(2, dim=1)
      gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
      return gradient_penalty