#!/usr/bin/env python3
"""
Script to install environment requirements from pip_env.txt
Handles version conflicts and compatibility issues for RL packages.
"""

import subprocess
import sys
import os
import re
from pathlib import Path
from typing import List, Tuple, Dict


def run_command(command: str, description: str = "", ignore_errors: bool = False) -> bool:
    """Run a command and handle errors."""
    print(f"{'=' * 60}")
    print(f"Running: {command}")
    if description:
        print(f"Description: {description}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if ignore_errors:
            print("Continuing despite error...")
            return True
        return False


def check_pip_env_file() -> bool:
    """Check if pip_env.txt exists."""
    req_file = Path("pip_env.txt")
    if not req_file.exists():
        print(f"ERROR: pip_env.txt not found in current directory: {os.getcwd()}")
        print("Please ensure pip_env.txt exists with your requirements.")
        return False

    # Check if file has content
    with open(req_file, 'r') as f:
        content = f.read().strip()
        if not content:
            print("ERROR: pip_env.txt is empty")
            return False

    print(f"✅ Found pip_env.txt with {len(content.splitlines())} packages")
    return True


def parse_requirements(file_path: str) -> List[Tuple[str, str]]:
    """Parse requirements file and return list of (package, version) tuples."""
    requirements = []

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse package==version format
            if '==' in line:
                package, version = line.split('==', 1)
                requirements.append((package.strip(), version.strip()))
            else:
                # Handle packages without version specifiers
                requirements.append((line.strip(), None))

    return requirements


def identify_problematic_packages(requirements: List[Tuple[str, str]]) -> Dict[str, List[str]]:
    """Identify packages that commonly cause conflicts."""
    problematic = {
        'version_conflicts': [],
        'deprecated': [],
        'special_handling': []
    }

    for package, version in requirements:
        package_lower = package.lower()

        # Known version conflicts
        if package_lower in ['gym', 'gymnasium'] and version:
            if package_lower == 'gym' and version.startswith('0.25'):
                problematic['version_conflicts'].append(f"{package}=={version} (conflicts with stable-baselines3)")

        # Deprecated packages
        if package_lower in ['atari-py']:
            problematic['deprecated'].append(f"{package}=={version} (deprecated, use ale-py)")

        # Special handling needed
        if package_lower in ['torch', 'torchvision', 'torchaudio']:
            if '+cu' in version:
                problematic['special_handling'].append(f"{package}=={version} (CUDA version)")

        if package_lower in ['mujoco']:
            problematic['special_handling'].append(f"{package}=={version} (may need system dependencies)")

    return problematic


def create_modified_requirements(requirements: List[Tuple[str, str]], output_file: str) -> str:
    """Create a modified requirements file with compatibility fixes."""
    modified_reqs = []

    for package, version in requirements:
        package_lower = package.lower()

        # Apply compatibility fixes
        if package_lower == 'gym' and version and version.startswith('0.25'):
            # Replace gym 0.25.x with gymnasium
            print(f"⚠️  Replacing {package}=={version} with gymnasium==0.28.1")
            modified_reqs.append("gymnasium==0.28.1")
        elif package_lower == 'stable-baselines3':
            # Ensure compatible version
            print(f"⚠️  Updating {package}=={version} to stable-baselines3>=2.0.0")
            modified_reqs.append("stable-baselines3>=2.0.0")
        elif package_lower in ['torch', 'torchvision', 'torchaudio'] and version and '+cu' in version:
            # Handle CUDA versions - install CPU version if CUDA not available
            base_version = version.split('+')[0]
            print(f"⚠️  Installing CPU version of {package}: {package}=={base_version}")
            modified_reqs.append(f"{package}=={base_version}")
        elif package_lower == 'gym-minigrid':
            # Replace with newer minigrid
            print(f"⚠️  Replacing {package}=={version} with minigrid")
            modified_reqs.append("minigrid")
        else:
            # Keep original requirement
            if version:
                modified_reqs.append(f"{package}=={version}")
            else:
                modified_reqs.append(package)

    # Write modified requirements
    with open(output_file, 'w') as f:
        f.write('\n'.join(modified_reqs))

    return output_file


def install_torch_first(requirements: List[Tuple[str, str]]) -> bool:
    """Install PyTorch first to avoid conflicts."""
    torch_packages = ['torch', 'torchvision', 'torchaudio']
    torch_reqs = [(pkg, ver) for pkg, ver in requirements if pkg.lower() in torch_packages]

    if not torch_reqs:
        return True

    print("\n🔥 Installing PyTorch packages first...")

    # Install CPU version of PyTorch
    torch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
    success = run_command(torch_cmd, "Install PyTorch (CPU version)")

    if not success:
        print("⚠️  Failed to install PyTorch. Continuing anyway...")

    return success


def install_in_batches(requirements_file: str) -> bool:
    """Install requirements in batches to handle conflicts better."""
    print(f"\n📦 Installing requirements from {requirements_file}...")

    # Strategy 1: Try installing all at once
    print("\n🎯 Strategy 1: Installing all packages at once...")
    success = run_command(f"pip install -r {requirements_file}", "Install all requirements")

    if success:
        print("✅ All packages installed successfully!")
        return True

    print("❌ Batch installation failed. Trying individual installation...")

    # Strategy 2: Install packages individually
    print("\n🎯 Strategy 2: Installing packages individually...")
    failed_packages = []

    with open(requirements_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            print(f"\nInstalling package {line_num}: {line}")
            success = run_command(f"pip install '{line}'", f"Install {line}", ignore_errors=True)

            if not success:
                failed_packages.append(line)
                print(f"⚠️  Failed to install {line}")

    if failed_packages:
        print(f"\n❌ Failed to install {len(failed_packages)} packages:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        return False
    else:
        print("\n✅ All packages installed individually!")
        return True


def install_system_dependencies():
    """Install system dependencies if needed."""
    print("\n🔧 Checking for system dependencies...")

    # Check if we're on Linux and need OpenGL libraries
    if sys.platform.startswith('linux'):
        print("Linux detected. You may need to install system dependencies:")
        print("sudo apt-get update")
        print("sudo apt-get install -y python3-opengl xvfb")
        print("sudo apt-get install -y ffmpeg")

        user_input = input("Install system dependencies automatically? (y/n): ").lower().strip()
        if user_input == 'y':
            commands = [
                "sudo apt-get update",
                "sudo apt-get install -y python3-opengl xvfb ffmpeg"
            ]
            for cmd in commands:
                run_command(cmd, "Install system dependencies", ignore_errors=True)


def verify_installation() -> bool:
    """Verify that key packages can be imported."""
    print("\n" + "=" * 60)
    print("🔍 VERIFYING INSTALLATION")
    print("=" * 60)

    # Test imports based on the requirements we saw
    test_imports = [
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("matplotlib", "Matplotlib"),
        ("stable_baselines3", "Stable Baselines3"),
        ("gymnasium", "Gymnasium"),
        ("minigrid", "MiniGrid"),
        ("ale_py", "ALE (Atari)"),
        ("h5py", "HDF5"),
        ("tqdm", "TQDM"),
        ("wandb", "Weights & Biases"),
    ]

    failed_imports = []
    successful_imports = []

    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name}: OK")
            successful_imports.append(name)
        except ImportError as e:
            print(f"❌ {name}: FAILED - {e}")
            failed_imports.append(name)

    print(f"\n📊 Results: {len(successful_imports)} successful, {len(failed_imports)} failed")

    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        print("These packages may need manual installation or have dependency conflicts.")
    else:
        print("\n🎉 All key packages imported successfully!")

    return len(failed_imports) == 0


def main():
    """Main installation script."""
    print("🚀 Installing RL Environment from pip_env.txt")
    print("=" * 60)

    # Check if pip_env.txt exists
    if not check_pip_env_file():
        return 1

    # Parse requirements
    print("\n📋 Parsing requirements...")
    requirements = parse_requirements("pip_env.txt")
    print(f"Found {len(requirements)} packages to install")

    # Identify problematic packages
    print("\n🔍 Analyzing requirements for potential conflicts...")
    problems = identify_problematic_packages(requirements)

    for category, packages in problems.items():
        if packages:
            print(f"\n⚠️  {category.replace('_', ' ').title()}:")
            for pkg in packages:
                print(f"  - {pkg}")

    # Create backup
    backup_file = "pip_env.txt.backup"
    if not Path(backup_file).exists():
        run_command("cp pip_env.txt pip_env.txt.backup", "Create backup")

    # Upgrade pip first
    print("\n⬆️  Upgrading pip...")
    run_command("python -m pip install --upgrade pip", "Upgrade pip")

    # Create modified requirements with fixes
    print("\n🛠️  Creating modified requirements with compatibility fixes...")
    modified_file = "pip_env_modified.txt"
    create_modified_requirements(requirements, modified_file)

    # Install system dependencies if needed
    install_system_dependencies()

    # Install PyTorch first
    install_torch_first(requirements)

    # Install remaining packages
    success = install_in_batches(modified_file)

    # Clean up temporary file
    if Path(modified_file).exists():
        os.remove(modified_file)

    # Verify installation
    print("\n🧪 Verifying installation...")
    verification_success = verify_installation()

    # Final summary
    print("\n" + "=" * 60)
    if success and verification_success:
        print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
        print("\nYou can now run your training script:")
        print(
            "python train.py --env_name minigrid-crossing-stochastic --ae_model_type ae --latent_dim 256 --mf_steps 1000000 --ae_recon_loss --ae_er_train")
    elif success:
        print("⚠️  INSTALLATION COMPLETED WITH WARNINGS")
        print("Some packages may have import issues. Check the verification results above.")
    else:
        print("❌ INSTALLATION FAILED")
        print("Some packages could not be installed. Check the error messages above.")

    print("\n💡 Tips:")
    print("- If you encounter CUDA issues, the script installed CPU versions")
    print("- For MuJoCo, you may need additional system setup")
    print("- Check the backup file pip_env.txt.backup if you need to restore")
    print("=" * 60)

    return 0 if (success and verification_success) else 1


if __name__ == "__main__":
    sys.exit(main())