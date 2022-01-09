import secrets
import string

def get_time_elapsed_ms(start_frame: int, end_frame: int, fps: float):
    return 1000.0 * (end_frame - start_frame) / fps

    
def get_random_string(length: int = 10, alphabet = string.ascii_letters + string.digits):
    return ''.join([secrets.choice(alphabet) for _ in range(length)])
