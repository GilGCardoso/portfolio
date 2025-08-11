import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Add parent directory to path so we can import our module
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(parent_dir)


# Check GPU availability
def _cuda_available() -> bool:

    logger.info("PyTorch version: %s", torch.__version__)   
    has_cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s", has_cuda)

    if has_cuda:
        try:
            name = torch.cuda.get_device_name()
        except Exception:
        logger.info("CUDA available: %s", name)
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        logger.info("GPU Memory: %.1f GB", props.total_memory / 1e9)

    else:
        logger.info("No CUDA GPU detected; using CPU.")

    return has_cuda


def compute_structure_factor_auto(self, *args, backend: str = "auto", **kwargs):
     """
    Dispatch to GPU or CPU implementation based on availability.
    Args/kwargs are passed through to the selected implementation.

    backend: "auto" | "gpu" | "cpu"
    """
    
    use_gpu = (backend == "gpu") or (backend == "auto" and _cuda_available())

    if use_gpu:
        from structure_factor_torch import StructureFactorCalculator_gpu
        GPU_calc= StructureFactorCalculator_gpu()

        return GPU_calc.calculate_structure_factor_and_save(*args, **kwargs)

    except Exception as e:
        logger.warning("GPU path failed (%s); falling back to CPU.", e)

    else:
        from structure_factor import StructureFactorCalculator_cpu
            CPU_calc = StructureFactorCalculator()
            return CPU_calc.calculate_structure_factor_and_save(*args, **kwargs)







    
        
     



