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

mesh_in=$1
grid_out=$2
external_root=$3
spacing=$4

outdir="$(dirname "${mesh_in}")"

gaps=${external_root}/gaps/bin/x86_64

mesh_orig=${outdir}/mesh_orig_tmp.${mesh_in##*.}
cp $mesh_in $mesh_orig

${gaps}/msh2df $mesh_orig $grid_out -estimate_sign -bbox -0.5 -0.5 -0.5 0.5 0.5 0.5 -spacing $spacing -border 0 -v

rm ${mesh_orig}
