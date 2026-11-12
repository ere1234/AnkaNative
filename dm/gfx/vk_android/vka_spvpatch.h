// SPIR-V rewrites made by the replay at pipeline time (vka_spvpatch.c). Plain C, no Vulkan
#pragma once
#include <stddef.h>
#include <stdint.h>
// The vertex module w (nw words) with the vec4 float Input variables at the locations of umask retyped as uvec4 and those of smask as ivec4, every load of them bit-cast back
// to float: the shader then sees the integer register bits Maxwell's fetch gives an integer format. A malloc'd copy (*out_nw words), or NULL with the reason in err
uint32_t *vka_spv_vin_retype(const uint32_t *w, size_t nw, uint32_t umask, uint32_t smask, size_t *out_nw, char *err, size_t errn);
// Subgroup operations and built-ins rewritten as subgroups of one invocation, for a stage the device runs none in (vka_spvpatch.c). A malloc'd copy (*out_nw words,
// *changed = 1), NULL with err empty when the module has none, NULL with the reason in err when it cannot be rewritten
uint32_t *vka_spv_subgroup_single(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn);
// A geometry shader's memory writes, counted (the return; -1 with err on failure) and, with out and a count above 0, the module with each of them made only by the run of
// the shader that first claims its primitive in the storage buffer at set 0 binding (zeroed for each draw, a word per primitive), malloc'd (*out_nw words): vka_spvpatch.c
int vka_spv_gs_claim(const uint32_t *w, size_t nw, uint32_t binding, uint32_t **out, size_t *out_nw, char *err, size_t errn);
// A pixel shader's float colour outputs at the locations of umask / smask as unsigned / signed integers of their bits (each store bit-cast), for integer attachments: malloc'd
// (*out_nw words), NULL with the reason in err (vka_spvpatch.c)
uint32_t *vka_spv_fout_retype(const uint32_t *w, size_t nw, uint32_t umask, uint32_t smask, size_t *out_nw, char *err, size_t errn);
// NVN2 task and mesh programs as compute shaders, for a device without VK_EXT_mesh_shader (vka_spvmesh.c; the replay: vka_draw.c "Mesh draws as compute").
// The storage buffer at rec_binding (words): a header the replay's passes write (selftest/meshemu.comp), the dispatch and draw arguments, then a record per task workgroup
// (its count of mesh workgroups, at most VKA_ME_MAXM, then its task memory), then the map (one word per mesh workgroup: its task | its index among that task's << 16)
#define VKA_ME_S 0          // words of one record: 1 + the task's memory words
#define VKA_ME_MAXM 1       // the most mesh workgroups one task workgroup launches (the rest are dropped; the map keeps the index in 16 bits)
#define VKA_ME_MODE 2       // 0: a mesh program alone (workgroup w is mesh workgroup w), 1: with a task (the map)
#define VKA_ME_MAPBASE 3    // the map's first word
#define VKA_ME_MAXSLOTS 4   // the most mesh workgroups the output buffer has slots for (one round)
#define VKA_ME_TASKS 5      // task workgroups (the prep pass)
#define VKA_ME_TOTAL 6      // mesh workgroups (the prep or scan pass)
#define VKA_ME_BASE 7       // the first mesh workgroup of this round: a mesh pass's workgroup w is mesh workgroup VKA_ME_BASE + w, its slot w
#define VKA_ME_MOST 8       // (these three outlive the draws: diagnostics) the most mesh workgroups one draw asked for
#define VKA_ME_OVER 9       // draws that asked for more mesh workgroups than their rounds hold (the rest dropped)
#define VKA_ME_TASK_OVER 10 // draws of more task workgroups than the records hold (the rest dropped)
#define VKA_ME_ARGS_TASK 16 // VkDispatchIndirectCommand of the task pass
#define VKA_ME_ARGS_MESH 20 // VkDispatchIndirectCommand of the mesh pass
#define VKA_ME_ARGS_DRAW 24 // VkDrawIndirectCommand of the draw
#define VKA_ME_REC0 32      // record t at VKA_ME_REC0 + t * S
typedef struct {
    int task;                               // a task program (else a mesh program)
    uint32_t threads;                       // invocations of a workgroup
    uint32_t payload_words;                 // its task memory's words (a mesh program's only when it reads it, else 0)
    uint32_t max_v, max_p;                  // mesh: the most vertices / primitives one workgroup outputs
    uint32_t nv, np, vloc[32], ploc[32];    // mesh: the per-vertex / per-primitive generic locations, in the slot's order (after the position / the indices)
    uint32_t slot_vec4;                     // mesh: 16-byte words of one workgroup's slot: max_v x (1 + nv) + max_p x (1 + np)
    uint32_t shared_bytes;                  // workgroup memory once the task memory and the outputs are in it
} vka_spv_mesh_info;
// The task or mesh module w as a GLCompute module (malloc'd, *out_nw words; NULL with the reason in err). A task reads the header and writes its record at rec_binding
// (a count of 0 when it returns without OpEmitMeshTasksEXT); a mesh program reads its task's record and the map there and writes its slot at out_binding, slot = its
// workgroup's index in the round's 1D dispatch
uint32_t *vka_spv_mesh_compute(const uint32_t *w, size_t nw, uint32_t rec_binding, uint32_t out_binding, vka_spv_mesh_info *info, size_t *out_nw, char *err, size_t errn);
// The vertex shader that draws a mesh program's slots (its info) from the storage buffer at out_binding as a triangle list (malloc'd, *out_nw words; NULL when the info
// is not a mesh program's)
uint32_t *vka_spv_mesh_vs(const vka_spv_mesh_info *info, uint32_t out_binding, size_t *out_nw);
// A pixel shader's PerPrimitiveEXT inputs as Flat ones, without the mesh shading capability (malloc'd, *out_nw words, *changed = 1); NULL with err empty when it has none
uint32_t *vka_spv_fs_mesh_flat(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn);
