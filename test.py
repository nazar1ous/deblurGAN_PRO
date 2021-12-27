from pytorch_lightning.metrics.regression import PSNR, SSIM
from tqdm import tqdm
import numpy as np
import torch
from src.lightning_classes.lightning_deblur_SRGAN import LightningModule


def test_model_inference_time(model, input_tensor, warmup_num=100, step_num=1000):
    model.cuda()
    for i in range(warmup_num):
        with torch.no_grad():
            input_tensor_ = torch.randn(*input_tensor.shape).cuda()
            _ = model(input_tensor_)

    timings = []
    for i in range(step_num):
        with torch.no_grad():
            input_tensor_ = torch.randn(*input_tensor.shape).cuda()
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            starter.record()

            _ = model(input_tensor_)

            ender.record()
            torch.cuda.synchronize()
            inference_time = starter.elapsed_time(ender)
            timings.append(inference_time)
    timings = np.array(timings)

    return timings


def test_model_metrics(model, test_dataloader):
    model.cuda()
    psnr_metric = PSNR(data_range=2.0)
    ssim_metric = SSIM(data_range=2.0)
    ssim_metrics = []
    psnr_metrics = []

    for batch in tqdm(test_dataloader):
        blurred, gt = batch
        gt = gt.cuda()
        blurred = blurred.cuda()

        output = model(blurred)
        ssim_value = ssim_metric(pred=output.clone().detach(),
                                 target=gt.clone().detach())
        psnr_value = psnr_metric(pred=output.clone().detach(),
                                 target=gt.clone().detach())
        ssim_metrics.append(ssim_value.item())
        psnr_metrics.append(psnr_value.item())

    return np.array(psnr_metrics), np.array(ssim_metrics)


if __name__ == "__main__":
    lm = LightningModule()
    weight_path = ""
    lm = lm.load_from_checkpoint(weight_path)
    lm.setup(stage="test")
    model = lm.generator
    test_dataloader = lm.test_dataloader()

