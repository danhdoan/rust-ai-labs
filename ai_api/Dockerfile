# ===============================================================================
# STAGE 1: Build Rust binary
# ===============================================================================
FROM rust:1.84 as builder

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    cmake \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Download & extract LibTorch
ENV LIBTORCH_ZIP=libtorch.zip
ENV LIBTORCH_ZIP_LINK=https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu118.zip
RUN wget -O ${LIBTORCH_ZIP} ${LIBTORCH_ZIP_LINK}  \
    && unzip ${LIBTORCH_ZIP} -d /usr/local/ \
    && rm ${LIBTORCH_ZIP}

# Set environment variables for LibTorch
ENV LIBTORCH=/usr/local/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Copy project files
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build the release binary
RUN cargo build --release

# ===============================================================================
# STAGE 2: Create minimal runtime image
# ===============================================================================
FROM debian:bookworm-slim

# Set the working directory
WORKDIR /app

ENV LIBTORCH=/usr/local/libtorch
ENV LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH

# Copy LibTorch runtime files
COPY --from=builder /usr/local/libtorch /usr/local/libtorch

# Copy the compiled binary from the builder stage
COPY --from=builder /app/target/release/ai_api .

# Copy ONNX model files
COPY models models

# ===============================================================================
# STAGE 3: Run application
# ===============================================================================
# Expose API port
EXPOSE 8000

# Run the AI API
CMD ["./ai_api"]
