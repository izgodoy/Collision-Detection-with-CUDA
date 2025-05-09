### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 30f005c8-2707-11f0-22fa-6b47eb7e2b11
begin
	using CUDA
	# set up pseudorandom behavior for reproducibility
	using Random
	rng = Xoshiro(123)
	Random.seed!(rng)
end

# ╔═╡ 4ba264f3-6f3d-4347-aa9c-f3fb02db9901
begin
	using Base.Threads
	using Distributed
	using Flatten
end

# ╔═╡ eb3c03f6-b86d-4371-ad7a-6b49fd96d403
md"""
# Broad-Phase Collision Detection with CUDA
"""

# ╔═╡ 10650c92-05ab-4b13-9d55-7247636f9689
md"""
## Generating spheres
First, write a function to generate n d-dimensional spheres with random positions and radii.
"""

# ╔═╡ 16f53307-0200-4fbf-892b-3a18ea7478ce
begin
	function generate_sphere_data(n::Int, d::Int; radius_range=(0.1, 1.0))
    	minr, maxr = radius_range
    	spheres = Dict{String,
					   NamedTuple{(:position, :radius),
					   Tuple{Vector{Float64},
					   Float64}}}()
    	for i in 1:n
        	key = "obj$i"
        	position = rand(d)
        	radius = minr + (maxr - minr) * rand()  # scale rand() to [minr, maxr]
        	spheres[key] = (position=position, radius=radius)
    	end
    	return spheres
	end

	# test an example
	sphere_dict = generate_sphere_data(5, 3)
end

# ╔═╡ 6891cf87-87b2-4b18-b330-43ca4588f592
md"""
# d-Dimensional Implementation
"""

# ╔═╡ 52008173-26eb-4347-880a-2f7247a07cf8
md"""
## Constructing the Cell ID Array
### Spatial Hashing
Spatial hashing maps a set of coordinates to a discrete cell ID so that nearby objects fall into the same or neighboring cells for fast lookups.

Note that each dimension has a different maximum grid resolution using a 32-bit hash given by ``2^{(\frac{32}{d})}``. For ``d=2``, the grid can be up to 65,536 x 65,536, and for ``d=3``, the grid can be up to 1,024 x 1,024 x 1,024.
"""

# ╔═╡ 0b53b1e1-9d55-4e69-b1f1-9a663bc94607
begin
	function spatial_hash(pos::AbstractVector{<:Float64}, cellsize::Real)
	    d = length(pos)
	    bits_per_dim = 32 ÷ d # ensure total fits in 32 bits
	    max_cells = 1 << bits_per_dim
	
	    @assert d * bits_per_dim <= 32 "Too many dimensions to fit in 32-bit hash"
	
	    hash = UInt32(0)
	    for i in 1:d
	        cell_index = UInt32(clamp(floor(Int, pos[i] / cellsize),
									  0, max_cells - 1))
	        shift = bits_per_dim * (i - 1)
	        hash |= cell_index << shift
	    end
	    return hash
	end

	# create a 3D example to test with
	pos = [2.6, 3.4, 7.9]
	radius = 0.8
	cellsize = 1.0
	
	hash = spatial_hash(pos, cellsize)
	hash_hex = string(hash, base = 16) # Convert to hex
	println("Hash: decimal = ", hash, ", hex = 0x", hash_hex)
end

# ╔═╡ bdbc2f91-505d-4943-84bc-530d690879b6
md"""
### Getting the "Home" Cell ID
The home cell of an object is the cell in which its centroid resides. An object can thus only have one home cell. We'll do this as part of the function below that finds neighboring cell hashes given an object position.
"""

# ╔═╡ 93b0edc2-fdb1-4aa2-83ef-a1004cf02f37
md"""
### Getting "Phantom" Cell IDs
Phantom cells, or P-cells, are cells that are intersected by an object but are not its home cell. We can search a cell's neighbors to find these. A cell can intersect a maximum of ``2^d - 1`` P-cells.
"""

# ╔═╡ 5b3fb5d4-34a2-4943-8aae-390d55d2b298
begin
	function neighboring_hashes(pos::AbstractVector{<:Real}, radius::Real, cellsize::Real)
	    d = length(pos)
	    bits_per_dim = 32 ÷ d
	    max_cells = 1 << bits_per_dim
	
	    # compute home cell index
	    home_idx = [floor(Int, pos[i] / cellsize) for i in 1:d]
	
	    neighbors = UInt32[]
	
	    # function to convert grid index to spatial hash
	    function index_to_hash(index)
	        # clamp to grid bounds and convert to cell position
	        clamped = clamp.(index, 0, max_cells - 1)
	        cell_pos = [i * cellsize for i in clamped]
	        return spatial_hash(cell_pos, cellsize)
	    end
	
	    # home cell is always the first index pushed to neighbor array
	    push!(neighbors, index_to_hash(home_idx))
	
	    # Face-adjacent neighbors (+-1 in one dimension at a time)
	    for i in 1:d
	        for offset in (-1, 1)
	            neighbor_idx = copy(home_idx)
	            neighbor_idx[i] += offset
	            push!(neighbors, index_to_hash(neighbor_idx))
	        end
	    end

		unique_neighbors = unique(neighbors)

		# if the object has fewer than 2^d neighboring cells, the extra cell IDs
		# are marked as 0xffffffff to indicate they are not valid
		padded_neighbors = vcat(
            unique_neighbors[1:min(end, 2^d)],
            fill(0xffffffff, max(0, 2^d - length(unique_neighbors)))
        )
	
	    return padded_neighbors
	end

	# test with the same example sphere from above	
	hashes = neighboring_hashes(pos, radius, cellsize)
	println("Overlapping cell hashes: ", hashes)
end

# ╔═╡ f03c6838-a686-4f62-b09b-2cce7c1e6f41
md"""
### Parallelizing the Generation of Cell IDs
Here, we take advantage of native Julia parallelization by calling `@threads`.
"""

# ╔═╡ 73717366-9ebe-483e-a683-8a35cdf19f37
function construct_cellIDs(positions::Vector{<:AbstractVector{<:Real}},
                         radii::Vector{<:Real},
                         cellsize::Real)
	d = length(positions[1])
    n = length(positions)
    cellID_arr = Array{UInt32}(undef, n * 2^d)
	objID_arr = Array{Int32}(undef, n)

    @threads for i in 1:n
        pos = positions[i]
        radius = radii[i]
        neighbors = neighboring_hashes(pos, radius, cellsize)

        offset = (i - 1) * 2^d
        for j in 1:2^d
            idx = offset + j
            cellID_arr[idx] = neighbors[j]
        end
        objID_arr[i] = i
    end

    return cellID_arr, objID_arr
end

# ╔═╡ 3751925f-e7c3-4763-9ddf-622f5d139c27
begin
	# testing the above function, assuming cellsize is still 0.8
	positions = [Vector(rand(3) * 10) for _ in 1:100_000]
	radii = rand(0.5:0.1:1.0, 100_000)

	cellID_arr, objID_arr = construct_cellIDs(positions, radii, cellsize)
	println("First object's entries from cell ID array: ", cellID_arr[1:min(8,end)])
	println("First 5 Object ID Array entries: ", objID_arr[1:min(5,end)])
end

# ╔═╡ b538ef69-8306-4d77-a58f-d7915113bb03
md"""
### Using Multithreading to Count Generated Cell IDs
While constructing the cell ID array, we also want to calculate the total number of cell IDs that get generated for later use in handling the potential collision cells. This involves 3 steps:
1. Each thread computes its own count of generated cell IDs.
2. Each thread block performs a reduction (sum) over its threads’ counts.
3. Then, the block leader (e.g. thread 0) writes the sum to a global array indexed by block ID.
"""

# ╔═╡ d3cd858f-4e41-478d-94e0-d0cadaa3db0b
md"""
### Single-Thread Cell ID Counting Kernel
"""

# ╔═╡ 03b11448-ebe3-4a34-8666-c133bee072e5
function generate_cell_ids_kernel(positions, radii, cell_ids, counts, cellsize)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i > length(positions)
        return
    end

    pos = positions[i]
    r = radii[i]

    # generate cell IDs for each object
    ids = neighboring_hashes(pos, r, cellsize)

    offset = counts[i]  # important: precompute prefix sum before launch
    for j in 1:length(ids)
        cell_ids[offset + j] = ids[j]
    end

    counts[i] = length(ids)
    return
end

# ╔═╡ a014f305-95d0-4331-8539-2f871de2dcac
md"""
### Single-Block Sum Kernel
"""

# ╔═╡ 2f60a5fd-8685-433a-b8c5-bebcbc741996
function count_per_block_kernel(counts, counts_per_block)
	shared = @cuStaticSharedMem(Int32, 256)  # Assumes block size = 256
    tid = threadIdx().x
    bid = blockIdx().x
    idx = (bid - 1) * blockDim().x + tid

    # use shared memory for block reduction
    shared[tid] = (idx <= length(counts)) ? counts[idx] : 0
    sync_threads()

    stride = blockDim().x ÷ 2
    while stride ≥ 1
        if tid ≤ stride && tid + stride ≤ blockDim().x
            local_counts[tid] += local_counts[tid + stride]
        end
        sync_threads()
        stride ÷= 2
    end

    if tid == 1
        counts_per_block[bid] = local_counts[1]
    end
end

# ╔═╡ 26075747-9adf-4a53-ad42-8b349a32d871
md"""
### A CPU-Fallback Prefix Sum
"""

# ╔═╡ 73df3e31-3893-4f93-85a5-7a3c95f84d69
function prefix_sum!(input::Vector{Int}, output::Vector{Int})
    @assert length(input) == length(output)
    acc = 0
    for i in 1:length(input)
        acc += input[i]
        output[i] = acc
    end
    return output
end

# ╔═╡ 0a63b76f-f45d-4a27-bbe8-64edb177f7d9
md"""
### Launching From Host
"""

# ╔═╡ 0c825de6-0ef3-4fdd-8265-2e13c323b370
function host_counting(positions::Vector, radii::Vector, cellsize::Real)
	d = length(positions[1])
	num_threads = 256
	num_blocks = cld(length(positions), num_threads)
	
	# allocate on GPU
	cell_ids = CUDA.zeros(UInt32, 2^d)
	counts = CUDA.zeros(Int32, length(positions))
	counts_per_block = CUDA.zeros(Int32, num_blocks)
	
	# launch first kernel
	positions_gpu = CuVector(positions)
	radii_gpu = CuVector(radii)

	@cuda threads=num_threads blocks=num_blocks generate_cell_ids_kernel(
	    positions_gpu, radii_gpu, cell_ids, counts, cellsize
	)

	# precompute prefix sum
	prefix_sum!(counts, offsets)
	
	# launch second kernel
	@cuda threads=num_threads blocks=num_blocks count_per_block_kernel(
	    counts, counts_per_block
	)
end

# ╔═╡ 781785ab-f248-4750-a78d-e02e032d969b
begin
	# testing parallelized counting
	host_counting(positions, radii, cellsize)
end

# ╔═╡ 9f17e760-118f-48d3-8da9-a70dd4d09081
md"""
## Sorting the Cell ID Array
### Parallel Radix Sort
We want elements with the same cell ID to be sorted by cell ID type (H before P), because this simplifies the collision cell processing, where we’ll need to iterate over all H cell IDs against the total of all H and P cell IDs. To do this, we need a stable sort algorithm that guarantees identical cell IDs remain in the same order as before sorting. We’ll elect to use a parallelized radix sort, which is easier said than done.

A basic radix sort works by sorting B-bit keys by groups of L bits within them in successive passes. For our purposes, we will sort 32-bit cell IDs. If, for example, we choose to separate them into 8-bit groups, we’ll make four passes. A radix sort happens to sort low-order bits before higher-order bits, which is exactly what we need for our implementation.

"Each sorting pass of a radix sort...masks off all the bits of each key except for the currently active set of L bits, and then it tabulates the number of occurrences of each of the 2^L possible values that result from this masking. The resulting array of radix counters is then processed to convert each value into offsets from the beginning of the array. In other words, each value of the table is replaced by the sum of all its preceding values; this operation is called a prefix sum. Finally, these offsets are used to determine the new positions of the keys for this pass, and the array of keys is reordered accordingly." (NVIDIA Docs)
"""

# ╔═╡ 12a15f7b-bc23-4809-b360-bef010b67857
md"""
### Phase 1: Setup and Tabulation

"""

# ╔═╡ ef58c4b1-73b4-457a-ac2d-4d25a4fd3eb3
function encode_keys_kernel!(
    keys::CuDeviceVector{UInt64},
    cell_ids::CuDeviceVector{UInt32},
    k::Int  # number of cell IDs per object = 2^d
	)
	
    idx = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if idx > length(cell_ids)
        return
    end

    offset_in_object = (idx - 1) % k
    type_bit = offset_in_object == 0 ? UInt64(0) : UInt64(1)

    keys[idx] = UInt64(cell_ids[idx]) << 1 | type_bit

	return nothing
end

# ╔═╡ c7efa378-5754-4e94-85bb-ee4cae125c3f
function flags_kernel!(flags, keys_in, bit, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        flags[i] = (keys_in[i] >> bit) & 1
    end
end

# ╔═╡ be4bc9d9-5188-4f2f-a72a-89a216d65531
function scatter_kernel!(flags, scanned, total_zeros, keys_in, vals_in, keys_out, vals_out, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        flag = flags[i]
        pos = flag == 0 ? (i - scanned[i]) : (total_zeros + scanned[i] - 1)

        keys_out[pos + 1] = keys_in[i]
        vals_out[pos + 1] = vals_in[i]
    end
end

# ╔═╡ b9e300e9-68c0-4a0d-a98a-c5b87c0bbc05
function radix_sort_pass!(
    keys_in::CuDeviceVector{UInt64},
    vals_in::CuDeviceVector{Int32},
    keys_out::CuDeviceVector{UInt64},
    vals_out::CuDeviceVector{Int32},
    bit::Int
)
    n = length(keys_in)

    # compute bit flags (0 or 1)
    flags = CUDA.zeros(Int32, n)
	@cuda threads=256 blocks=cld(n, 256) flags_kernel!(flags, keys_in, bit, n)

    # inclusive scan (prefix sum of flags)
    scanned = CUDA.zeros(Int32, n)
    CUDA.prefixsum!(scanned, flags)

    # total number of zeros = n - scanned[end]
    total_ones = scanned[end]
    total_zeros = n - total_ones

    # scatter based on bit
    flags = CUDA.zeros(Int32, n)
	scanned = CUDA.zeros(Int32, n)
	total_zeros = n - total_ones
	
	# Launch the kernel
	@cuda threads=256 blocks=cld(n, 256) scatter_kernel!(flags, scanned, total_zeros, keys_in, vals_in, keys_out, vals_out, n)

    return nothing # free extra memory
end

# ╔═╡ 789800e9-91ca-4bc7-af7b-9fdaeac06d02
function full_radix_sort!(keys::CuDeviceVector{UInt64}, vals::CuDeviceVector{Int32})
    nbits = 64
    tmp_keys = similar(keys)
    tmp_vals = similar(vals)

    from_keys = keys
    from_vals = vals
    to_keys = tmp_keys
    to_vals = tmp_vals

    for bit in 0:nbits-1
        radix_sort_pass!(from_keys, from_vals, to_keys, to_vals, bit)
        from_keys, to_keys = to_keys, from_keys
        from_vals, to_vals = to_vals, from_vals
    end

    if from_keys != keys
        copyto!(keys, from_keys)
        copyto!(vals, from_vals)
    end
end

# ╔═╡ 0bf81009-8d6d-4acf-8944-5232a299d624
function host_radix_kernel(
	d::Int,
	cellIDs::Vector{UInt32},
	objectIDs::Vector{Int32}
)
	k = 2^d
	n = length(cellID_arr) ÷ k
	keys = CUDA.zeros(UInt64, n * k)

	cellID_arr_gpu = CuArray(cellIDs)
	objID_arr_gpu = CuArray(objectIDs)
	
	@cuda threads=256 blocks=cld(n * k, 256) encode_keys_kernel!(keys, cellID_arr_gpu, k)

	full_radix_sort!(keys, objID_arr_gpu)
end

# ╔═╡ 97f35301-8731-4836-a283-3e625c89323e
begin
	#test above functionality
	host_radix_kernel(3, cellID_arr, objID_arr)
end

# ╔═╡ 88c25cc9-c055-433e-aaba-5c9e1d2c75df
md"""
# Sources
[Julia Documentation](https://docs.julialang.org/en/v1/)

[CUDA.jl Documentation](https://cuda.juliagpu.org/stable/)

[NVIDIA Docs on Broad Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Distributed = "8ba89e20-285c-5b6f-9357-94700520ee1b"
Flatten = "4c728ea3-d9ee-5c9a-9642-b6f7d7dc04fa"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
CUDA = "~5.7.3"
Flatten = "~0.4.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "b2f779d77b0ef63d192259779616c96e44c55295"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

    [deps.AbstractFFTs.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "f7817e2e585aa6d924fd714df1e2a84be7896c60"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.3.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "b5bb4dc6248fde467be2a863eb8452993e74d402"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.1.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixOpenCLExt = "OpenCL"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    OpenCL = "08131aa3-fb12-5dee-8b74-c09406e224a2"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.BFloat16s]]
deps = ["LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3b642331600250f592719140c60cf12372b82d66"
uuid = "ab4f0b2a-ad5b-11e8-123f-65d77653426b"
version = "0.5.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CUDA]]
deps = ["AbstractFFTs", "Adapt", "BFloat16s", "CEnum", "CUDA_Driver_jll", "CUDA_Runtime_Discovery", "CUDA_Runtime_jll", "Crayons", "DataFrames", "ExprTools", "GPUArrays", "GPUCompiler", "GPUToolbox", "KernelAbstractions", "LLVM", "LLVMLoopInfo", "LazyArtifacts", "Libdl", "LinearAlgebra", "Logging", "NVTX", "Preferences", "PrettyTables", "Printf", "Random", "Random123", "RandomNumbers", "Reexport", "Requires", "SparseArrays", "StaticArrays", "Statistics", "demumble_jll"]
git-tree-sha1 = "ee44b6eaaf518e8001ed6ed1d08a04662d63a242"
uuid = "052768ef-5323-5732-b1bb-66c8b64840ba"
version = "5.7.3"

    [deps.CUDA.extensions]
    ChainRulesCoreExt = "ChainRulesCore"
    EnzymeCoreExt = "EnzymeCore"
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.CUDA.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f69205592dbd3721a156245b6dd837206786a848"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.12.1+1"

[[deps.CUDA_Runtime_Discovery]]
deps = ["Libdl"]
git-tree-sha1 = "33576c7c1b2500f8e7e6baa082e04563203b3a45"
uuid = "1af6417a-86b4-443c-805f-a4643ffb695f"
version = "0.3.5"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "99f1c6b659c14bbb3492246791bb4928a40ceb84"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.16.1+0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.FieldMetadata]]
git-tree-sha1 = "c279c6eab9767a3f62685e5276c850512e0a1afd"
uuid = "bf96fef3-21d2-5d20-8afa-0e7d4c32a885"
version = "0.3.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Flatten]]
deps = ["ConstructionBase", "FieldMetadata"]
git-tree-sha1 = "d3541c658c7e452fefba6c933c43842282cdfd3e"
uuid = "4c728ea3-d9ee-5c9a-9642-b6f7d7dc04fa"
version = "0.4.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GPUArrays]]
deps = ["Adapt", "GPUArraysCore", "KernelAbstractions", "LLVM", "LinearAlgebra", "Printf", "Random", "Reexport", "ScopedValues", "Serialization", "Statistics"]
git-tree-sha1 = "eea7b3a1964b4de269bb380462a9da604be7fcdb"
uuid = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
version = "11.2.2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GPUCompiler]]
deps = ["ExprTools", "InteractiveUtils", "LLVM", "Libdl", "Logging", "PrecompileTools", "Preferences", "Scratch", "Serialization", "TOML", "TimerOutputs", "UUIDs"]
git-tree-sha1 = "b08c164134dd0dbc76ff54e45e016cf7f30e16a4"
uuid = "61eb1bfa-7361-4325-ad38-22787b887f55"
version = "1.3.2"

[[deps.GPUToolbox]]
git-tree-sha1 = "15d8b0f5a6dca9bf8c02eeaf6687660dafa638d0"
uuid = "096a3bc2-3ced-46d0-87f4-dd12716f4bfc"
version = "0.2.0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JuliaNVTXCallbacks_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "af433a10f3942e882d3c671aacb203e006a5808f"
uuid = "9c1d0b0a-7046-5b2e-a33f-ea22f176ac7e"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "80d268b2f4e396edc5ea004d1e0f569231c71e9e"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.34"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.LLVM]]
deps = ["CEnum", "LLVMExtra_jll", "Libdl", "Preferences", "Printf", "Unicode"]
git-tree-sha1 = "5fcfea6df2ff3e4da708a40c969c3812162346df"
uuid = "929cbde3-209d-540e-8aea-75f648917ca0"
version = "9.2.0"
weakdeps = ["BFloat16s"]

    [deps.LLVM.extensions]
    BFloat16sExt = "BFloat16s"

[[deps.LLVMExtra_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "4b5ad6a4ffa91a00050a964492bc4f86bb48cea0"
uuid = "dad2f222-ce93-54a1-a47d-0025e8a3acab"
version = "0.0.35+0"

[[deps.LLVMLoopInfo]]
git-tree-sha1 = "2e5c102cfc41f48ae4740c7eca7743cc7e7b75ea"
uuid = "8b046642-f1f6-4319-8d3c-209ddc03c586"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NVTX]]
deps = ["Colors", "JuliaNVTXCallbacks_jll", "Libdl", "NVTX_jll"]
git-tree-sha1 = "1a24c3430fa2ef3317c4c97fa7e431ef45793bd2"
uuid = "5da4648a-3479-48b8-97b9-01cb529c0a1f"
version = "1.0.0"

[[deps.NVTX_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2c7b791c1eba364e4a70aabdea4ddc1f5ca53911"
uuid = "e98f9f5b-d649-5603-91fd-7774390e6439"
version = "3.1.1+0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Random123]]
deps = ["Random", "RandomNumbers"]
git-tree-sha1 = "4743b43e5a9c4a2ede372de7061eed81795b12e7"
uuid = "74087812-796a-5b5d-8853-05524746bad3"
version = "1.7.0"

[[deps.RandomNumbers]]
deps = ["Random"]
git-tree-sha1 = "c6ec94d2aaba1ab2ff983052cf6a606ca5985902"
uuid = "e6cf234a-135c-5ec9-84dd-332b85af5143"
version = "1.6.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "1147f140b4c8ddab224c94efa9569fc23d63ab44"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.3.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "0feb6b9031bd5c51f9072393eb5ab3efd31bf9e4"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.13"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f57facfd1be61c42321765d3551b3df50f7e09f6"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.28"

    [deps.TimerOutputs.extensions]
    FlameGraphsExt = "FlameGraphs"

    [deps.TimerOutputs.weakdeps]
    FlameGraphs = "08572546-2f56-4bcf-ba4e-bab62c3a3f89"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"
weakdeps = ["LLVM"]

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.demumble_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6498e3581023f8e530f34760d18f75a69e3a4ea8"
uuid = "1e29f10c-031c-5a83-9565-69cddfc27673"
version = "1.3.0+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╟─eb3c03f6-b86d-4371-ad7a-6b49fd96d403
# ╠═30f005c8-2707-11f0-22fa-6b47eb7e2b11
# ╟─10650c92-05ab-4b13-9d55-7247636f9689
# ╠═16f53307-0200-4fbf-892b-3a18ea7478ce
# ╟─6891cf87-87b2-4b18-b330-43ca4588f592
# ╟─52008173-26eb-4347-880a-2f7247a07cf8
# ╠═0b53b1e1-9d55-4e69-b1f1-9a663bc94607
# ╟─bdbc2f91-505d-4943-84bc-530d690879b6
# ╟─93b0edc2-fdb1-4aa2-83ef-a1004cf02f37
# ╠═5b3fb5d4-34a2-4943-8aae-390d55d2b298
# ╟─f03c6838-a686-4f62-b09b-2cce7c1e6f41
# ╠═4ba264f3-6f3d-4347-aa9c-f3fb02db9901
# ╠═73717366-9ebe-483e-a683-8a35cdf19f37
# ╠═3751925f-e7c3-4763-9ddf-622f5d139c27
# ╟─b538ef69-8306-4d77-a58f-d7915113bb03
# ╟─d3cd858f-4e41-478d-94e0-d0cadaa3db0b
# ╠═03b11448-ebe3-4a34-8666-c133bee072e5
# ╟─a014f305-95d0-4331-8539-2f871de2dcac
# ╠═2f60a5fd-8685-433a-b8c5-bebcbc741996
# ╟─26075747-9adf-4a53-ad42-8b349a32d871
# ╠═73df3e31-3893-4f93-85a5-7a3c95f84d69
# ╟─0a63b76f-f45d-4a27-bbe8-64edb177f7d9
# ╠═0c825de6-0ef3-4fdd-8265-2e13c323b370
# ╠═781785ab-f248-4750-a78d-e02e032d969b
# ╟─9f17e760-118f-48d3-8da9-a70dd4d09081
# ╟─12a15f7b-bc23-4809-b360-bef010b67857
# ╠═ef58c4b1-73b4-457a-ac2d-4d25a4fd3eb3
# ╠═b9e300e9-68c0-4a0d-a98a-c5b87c0bbc05
# ╠═c7efa378-5754-4e94-85bb-ee4cae125c3f
# ╠═be4bc9d9-5188-4f2f-a72a-89a216d65531
# ╠═789800e9-91ca-4bc7-af7b-9fdaeac06d02
# ╠═0bf81009-8d6d-4acf-8944-5232a299d624
# ╠═97f35301-8731-4836-a283-3e625c89323e
# ╟─88c25cc9-c055-433e-aaba-5c9e1d2c75df
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
