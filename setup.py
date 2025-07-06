"""Setup script for FluxKontext Smart Assistant"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="forge-kontext-assistant",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Intelligent prompt generation for FLUX.1 Kontext in Forge WebUI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/forge-kontext-assistant",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "forge_kontext_assistant": [
            "configs/*.json",
            "configs/prompts/*.json"
        ]
    },
    entry_points={
        "console_scripts": [
            "kontext-assistant=forge_kontext_assistant.cli:main",
        ],
    },
)