### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 30f005c8-2707-11f0-22fa-6b47eb7e2b11
begin
	# remember to add and test CUDA using Pkg before use
	# set up random behavior
	using Random
	rng = Xoshiro(123)
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
Spatial hashing maps a set of coordinates (in this case, each sphere's centroid) to a discrete cell ID so that nearby objects fall into the same or neighboring cells for fast lookups.

Note that each dimension has a different maximum grid resolution using a 32-bit hash given by ``2^{(\frac{32}{d})}``. For ``d=2``, the grid can be up to 65,536 x 65,536, and for ``d=3``, the grid can be up to 1,024 x 1,024 x 1,024.
"""

# ╔═╡ 0b53b1e1-9d55-4e69-b1f1-9a663bc94607
begin
	function spatial_hash(pos::AbstractVector{<:Float64}, cellsize::Real)
	    d = length(pos)
	    bits_per_dim = 32 ÷ d # Ensure total fits in 32 bits
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

	# Test with a 3D example
	pos = [2.6, 3.4, 7.9]
	cellsize = 1.0
	hash = spatial_hash(pos, cellsize)
	println("Hash = ", hash)
end

# ╔═╡ 93b0edc2-fdb1-4aa2-83ef-a1004cf02f37
md"""
### Getting "Phantom" Cell IDs
Phantom cells are cells that are intersected by an object but are not its home cell. We can search a cell's neighbors to find these.
"""

# ╔═╡ efd1bbe4-10c9-49ef-b256-d3ea54094e88
begin
	function neighboring_hashes(pos::AbstractVector{<:Real}, radius::Real, cellsize::Real)
	    d = length(pos)
	    bits_per_dim = 32 ÷ d
	    max_cells = 1 << bits_per_dim

	    # Get index ranges for each dimension
	    ranges = [
	        floor(Int, (pos[i] - radius) / cellsize): floor(Int, (pos[i] + radius) / cellsize)
	        for i in 1:d
	    ]
	
	    hashes = UInt32[]
	    
	    # Recursive generator for multi-dimensional index tuples
	    function build_hash(index_tuple::Vector{Int}, dim::Int)
	        if dim > d
	            # Compute hash from the full index tuple
	            h = UInt32(0)
	            for i in 1:d
	                idx = clamp(index_tuple[i], 0, max_cells - 1)
	                h |= UInt32(idx) << ((i - 1) * bits_per_dim)
	            end
	            push!(hashes, h)
	            return
	        end
	        for val in ranges[dim]
	            index_tuple[dim] = val
	            build_hash(index_tuple, dim + 1)
	        end
	    end
	
	    build_hash(Vector{Int}(undef, d), 1)
	
	    return hashes
	end
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
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "fa3e19418881bf344f5796e1504923a7c80ab1ed"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"
"""

# ╔═╡ Cell order:
# ╟─eb3c03f6-b86d-4371-ad7a-6b49fd96d403
# ╠═30f005c8-2707-11f0-22fa-6b47eb7e2b11
# ╟─10650c92-05ab-4b13-9d55-7247636f9689
# ╠═16f53307-0200-4fbf-892b-3a18ea7478ce
# ╟─6891cf87-87b2-4b18-b330-43ca4588f592
# ╟─52008173-26eb-4347-880a-2f7247a07cf8
# ╠═0b53b1e1-9d55-4e69-b1f1-9a663bc94607
# ╟─93b0edc2-fdb1-4aa2-83ef-a1004cf02f37
# ╠═efd1bbe4-10c9-49ef-b256-d3ea54094e88
# ╟─88c25cc9-c055-433e-aaba-5c9e1d2c75df
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
