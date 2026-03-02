# check_cudnn_loaded.py
# Print the libcudnn shared libraries that are loaded in the current Python process.

def main():
    libs = set()
    with open("/proc/self/maps", "r") as f:
        for line in f:
            if "libcudnn" in line:
                libs.add(line.split()[-1])

    if not libs:
        print("No libcudnn* libraries found in /proc/self/maps (cuDNN not loaded in this process).")
        return

    print("Loaded cuDNN libraries:")
    for p in sorted(libs):
        print(p)

if __name__ == "__main__":
    main()
