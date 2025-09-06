FROM python:3.11-slim-bookworm AS builder

# Install build dependencies for OpenCV compilation
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        pkg-config \
        zlib1g-dev \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages (including OpenCV which may need compilation)
COPY app /app
RUN python -m pip install /app --extra-index-url https://www.piwheels.org/simple && \
    python -m pip install fastapi uvicorn requests apriltag && \
    python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless && \
    python -m pip install opencv-python==4.10.0.84 --extra-index-url https://www.piwheels.org/simple

# Final runtime stage
FROM python:3.11-slim-bookworm

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8000/tcp

# application version.  This should match the register_service file's version
LABEL version="0.0.1"

# Permissions for the container
# "Binds" section maps the host PC directories to the application directories
# "CpuQuota" and "CpuPeriod" limit to a single CPU core
LABEL permissions='\
{\
  "ExposedPorts": {\
    "8000/tcp": {}\
  },\
  "HostConfig": {\
    "Binds":[\
      "/usr/blueos/extensions/precision-landing/settings:/app/settings",\
      "/usr/blueos/extensions/precision-landing/logs:/app/logs"\
    ],\
    "ExtraHosts": [\
      "host.docker.internal:host-gateway"\
    ],\
    "PortBindings": {\
      "8000/tcp": [\
        {\
          "HostPort": ""\
        }\
      ]\
    }\
  }\
}'

LABEL authors='[\
    {\
        "name": "Randy Mackay",\
        "email": "rmackay9@yahoo.com"\
    }\
]'

LABEL company='{\
    "about": "ArduPilot",\
    "name": "ArduPilot",\
    "email": "rmackay9@yahoo.com"\
}'

LABEL readme='https://github.com/rmackay9/blueos-precision-landing/blob/main/README.md'
LABEL type="device-integration"
LABEL tags='[\
  "data-collection"\
]'
LABEL links='{\
        "source": "https://github.com/rmackay9/blueos-precision-landing"\
    }'
LABEL requirements="core >= 1.1"

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0"]
