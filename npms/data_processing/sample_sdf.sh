#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

watertight_mesh_in=$1
external_root=$2

outdir="$(dirname "${watertight_mesh_in}")"

gaps=${external_root}/gaps/bin/x86_64

# Step 2) Generate the near surface points:
${gaps}/msh2pts -near_surface -max_distance 0\.05 -num_points 300000 -v $watertight_mesh_in ${outdir}/samples_near.sdf

# Step 3) Generate the uniform points:
${gaps}/msh2pts -uniform_in_bbox -bbox -0\.5 -0\.5 -0\.5 0\.5 0\.5 0\.5 -npoints 100000 $watertight_mesh_in ${outdir}/samples_uniform.sdf

# Step 5) Generate the random surface points:
${gaps}/msh2pts -random_surface_points -num_points 100000 $watertight_mesh_in ${outdir}/samples_surface.pts