# Container Readiness

Run after self-hosting is approved. `preflight.md` proves the host has a GPU. This file
proves containers can use it.

Complete every check in this file before post-approval `list-model-profiles` or any large
image or model pull. Profile discovery may pull the selected NIM image, so storage and
cache checks are part of this gate.

## 1. Docker

Require all three commands to succeed:

```bash
docker version
docker info
docker compose version
```

`docker version` must show a reachable server, not only a client. Stop on daemon,
permission, or Compose-plugin errors. Report the failing layer. Do not silently add
`sudo` or change Docker configuration.

## 2. NVIDIA Container Toolkit

Follow the current
[NVIDIA Container Toolkit sample workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html).
Its Docker check is:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

This intentionally pulls a small image when it is not cached. Require the container output
to list the same intended GPU devices seen on the host. A working host `nvidia-smi` alone
does not pass this check.

If the command fails, stop before NIM, vLLM, or Riva pulls. Point to NVIDIA Container
Toolkit installation or Docker runtime configuration based on the actual error.

## 3. Storage and cache paths

Get Docker's storage root with:

```bash
docker info --format '{{.DockerRootDir}}'
```

Check free space on that filesystem and every host path that the generated
`compose.yaml` will mount. These include the locked NIM cache, Hugging Face cache, and
Jetson Thor Riva `model_repository` when applicable.

For each host-mounted cache:

1. resolve the exact path from the approved deployment source
2. create it only when it does not exist and the parent is approved
3. require it to be a directory and writable by the host account and container user
   required by the source launch command
4. check free space before pulling

Do not replace an existing cache, change ownership recursively, or delete files to make
space. Report the required path and available space when the check fails.

## 4. Credentials and registry

Reconfirm the required keys from `preflight.md` without printing them. Use the current
build.nvidia.com login instructions for NGC. A registry authentication failure is a hard
stop before model pulls.

## 5. Jetson Thor

When the routed platform is Jetson Thor, also require:

- the JetPack release supported by the current Riva ARM64 Quick Start
- NGC CLI authentication
- `hf auth whoami` access to the locked Nemotron repository
- a writable parent path with enough space for the external Riva `model_repository`

The repository itself is produced later by the one-time `riva_init.sh` step. Require it to
exist before starting the generated `riva` service, not before generating the project.

## Pass condition

Proceed to exact local profile discovery, `output-contract.md`, and the routed deployment
guide only when:

- Docker daemon and Compose are reachable
- a container can see every assigned NVIDIA GPU
- all required host-mounted paths are writable
- Docker storage and model caches have enough free space
- required registries and model sources can authenticate
- Jetson Thor prerequisites above pass when applicable

Record the commands, expected GPU assignment, and cache paths in the generated README.
