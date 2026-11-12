// NVN2 task and mesh programs on a device without VK_EXT_mesh_shader (plain C, no Vulkan; the replay side is vka_draw.c "Mesh draws as compute").
//
// The translator makes a task program a TaskEXT entry point and a mesh program a MeshEXT one (SpirvMeshEntry.cs): a wrapper "main" that calls the translated body
// between two workgroup barriers, its stage memory in a TaskPayloadWorkgroupEXT array of words, a mesh program's outputs in Output arrays (the position in a block of
// one vec4, one vec4 array per generic location, one per per-primitive location decorated PerPrimitiveEXT, the triangles' indices in PrimitiveTriangleIndicesEXT),
// written straight from the body and the wrapper. Samsung's RADV for Xclipse has no mesh stage (Ocarina of Time's task / mesh draws were skipped there), so the replay
// runs them as compute shaders and draws what they output with a vertex shader of its own:
//   vka_spv_mesh_compute: the module as a GLCompute one. Its task memory and outputs become workgroup memory (the code that writes them is left as it is); a task's
//     OpEmitMeshTasksEXT becomes a copy of its count and task memory into its record of the storage buffer at rec_binding; a mesh program starts by taking its task's
//     record and its own index among the workgroups that task launched (the map the scan pass wrote: VKA_ME_*), and ends by copying its outputs into its slot of the
//     storage buffer at out_binding (one slot per workgroup: max vertices x (position + generics), then max primitives x (indices + per-primitive values), 16 bytes each)
//   vka_spv_mesh_vs: the vertex shader that draws those slots as a triangle list, three vertices a primitive (gl_VertexIndex = (slot x max primitives + t) x 3 + corner):
//     each vertex fetched by its index, the per-primitive values given to all three vertices (flat)
//   vka_spv_fs_mesh_flat: a pixel shader's per-primitive inputs made flat ones, so it runs with a vertex shader
#include "vka_spvpatch.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

enum { OpName = 5, OpMemberName = 6, OpExtension = 10, OpMemoryModel = 14, OpEntryPoint = 15, OpExecutionMode = 16, OpCapability = 17,
       OpTypeVoid = 19, OpTypeBool = 20, OpTypeInt = 21, OpTypeFloat = 22, OpTypeVector = 23, OpTypeArray = 28, OpTypeRuntimeArray = 29, OpTypeStruct = 30, OpTypePointer = 32,
       OpTypeFunction = 33, OpConstant = 43, OpFunction = 54, OpFunctionEnd = 56, OpVariable = 59, OpLoad = 61, OpStore = 62, OpCopyMemory = 63, OpCopyMemorySized = 64,
       OpAccessChain = 65, OpInBoundsAccessChain = 66, OpPtrAccessChain = 67, OpDecorate = 71, OpMemberDecorate = 72, OpGroupDecorate = 74, OpGroupMemberDecorate = 75,
       OpCompositeConstruct = 80, OpCompositeExtract = 81, OpBitcast = 124, OpIAdd = 128, OpISub = 130, OpIMul = 132, OpUDiv = 134, OpUMod = 137, OpLogicalAnd = 167,
       OpSelect = 169, OpIEqual = 170, OpINotEqual = 171, OpULessThan = 176, OpShiftRightLogical = 194, OpBitwiseAnd = 199, OpControlBarrier = 224, OpLabel = 248,
       OpReturn = 253, OpExecutionModeId = 331, OpDecorateId = 332, OpDecorateString = 5632, OpMemberDecorateString = 5633, OpEmitMeshTasksEXT = 5294, OpSetMeshOutputsEXT = 5295 };
enum { SCInput = 1, SCOutput = 3, SCWorkgroup = 4, SCPrivate = 6, SCStorageBuffer = 12, SCTaskPayload = 5402 };
enum { DecBlock = 2, DecArrayStride = 6, DecBuiltIn = 11, DecFlat = 14, DecNonWritable = 24, DecLocation = 30, DecBinding = 33, DecDescriptorSet = 34, DecOffset = 35,
       DecPerPrimitive = 5271 };
enum { BIPosition = 0, BINumWorkgroups = 24, BIWorkgroupId = 26, BILocalInvocationIndex = 29, BIVertexIndex = 42, BIPrimPoint = 5294, BIPrimLine = 5295, BIPrimTriangle = 5296 };
enum { EMTask = 5364, EMMesh = 5365, EMGLCompute = 5, EMVertex = 0 };
enum { XMLocalSize = 17, XMLocalSizeId = 38, XMOutputVertices = 26, XMOutputPoints = 27, XMOutputLines = 5269, XMOutputPrimitives = 5270, XMOutputTriangles = 5298 };
enum { CapMeshShading = 5283 };

typedef struct { uint32_t *w; size_t n, cap; int bad; } buf_t;
static void bput(buf_t *b, uint32_t x) {
    if (b->bad) return;
    if (b->n == b->cap) { size_t c = b->cap ? 2 * b->cap : 1024; uint32_t *p = realloc(b->w, c * 4); if (!p) { b->bad = 1; return; } b->w = p; b->cap = c; }
    b->w[b->n++] = x;
}
static void bputn(buf_t *b, const uint32_t *x, size_t n) { for (size_t i = 0; i < n; i++) bput(b, x[i]); }
static void binst(buf_t *b, uint32_t op, const uint32_t *a, size_t n) { bput(b, (uint32_t)(n + 1) << 16 | op); bputn(b, a, n); }
#define I(b, op, ...) binst((b), (op), (const uint32_t[]){__VA_ARGS__}, sizeof((const uint32_t[]){__VA_ARGS__}) / 4)

// A string operand's words (the string, its NUL, the padding)
static uint32_t str_words(const uint32_t *w, size_t max) { for (size_t i = 0; i < max; i++) if (!(w[i] & 0xff000000u)) return (uint32_t)i + 1; return (uint32_t)max; }
static int str_is(const uint32_t *w, size_t max, const char *s) { size_t n = strlen(s); return (n + 4) / 4 <= max && !memcmp(w, s, n + 1); }

// ---------- vka_spv_mesh_compute ----------
typedef struct {
    uint32_t nb;                                   // the next new id
    uint32_t cval[256], cid[256]; int nc, over;    // new u32 constants (over: the table ran out)
    uint32_t t_u32, t_bool, t_v3u, t_v4u, t_v4f;   // types (the module's, or new: mk_*)
    uint32_t mk_bool, mk_v3u, mk_v4u, mk_v4f;
    uint32_t p_sb_u32, p_wg_u32, p_sb_v4u, p_wg_v4f, p_wg_v3u, p_in_v3u, p_in_u32, p_priv_v3u;
    uint32_t rta_u, st_rec, p_sb_rec, v_rec, rta_v4u, st_out, p_sb_out, v_out;
    uint32_t wid_new, lid_new, wid, lid;           // the workgroup ID input the new code reads (mesh: a new one; the module's becomes a private copy), the invocation index
    int full;                                      // the constant table is closed: a missing constant is an error
} gen_t;
static uint32_t cu(gen_t *g, uint32_t v) {
    for (int i = 0; i < g->nc; i++) if (g->cval[i] == v) return g->cid[i];
    if (g->full || g->nc == 256) { g->over = 1; return 0; }
    g->cval[g->nc] = v; g->cid[g->nc] = g->nb++; return g->cid[g->nc++];
}
static uint32_t rec_word(buf_t *b, gen_t *g, uint32_t idx_id) {   // a load of rec[idx]
    uint32_t p = g->nb++, v = g->nb++;
    I(b, OpAccessChain, g->p_sb_u32, p, g->v_rec, cu(g, 0), idx_id); I(b, OpLoad, g->t_u32, v, p);
    return v;
}
static void barrier(buf_t *b, gen_t *g) { I(b, OpControlBarrier, cu(g, 2), cu(g, 2), cu(g, 0x108)); }   // Workgroup scope, acquire-release of workgroup memory
// The element of this iteration: lid + k x threads, clamped below n when the iteration has invocations past it (they copy the last element again: the same value)
static uint32_t elem(buf_t *b, gen_t *g, uint32_t lidv, uint32_t k, uint32_t threads, uint32_t n) {
    uint32_t e = lidv;
    if (k) { e = g->nb++; I(b, OpIAdd, g->t_u32, e, lidv, cu(g, k * threads)); }
    if ((k + 1) * threads > n) { uint32_t lt = g->nb++, s = g->nb++; I(b, OpULessThan, g->t_bool, lt, e, cu(g, n)); I(b, OpSelect, g->t_u32, s, lt, e, cu(g, n - 1)); e = s; }
    return e;
}

typedef struct { uint32_t var, loc; } outv;
static int by_loc(const void *a, const void *b) { const outv *x = a, *y = b; return x->loc < y->loc ? -1 : x->loc > y->loc; }

uint32_t *vka_spv_mesh_compute(const uint32_t *w, size_t nw, uint32_t rec_binding, uint32_t out_binding, vka_spv_mesh_info *info, size_t *out_nw, char *err, size_t errn) {
#define FAIL(...) do { snprintf(err, errn, __VA_ARGS__); goto fail; } while (0)
    memset(info, 0, sizeof *info); err[0] = 0;
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3], ver = w[1];
    if (bound > (1u << 22)) { snprintf(err, errn, "an id bound of %u", bound); return NULL; }   // (a translated program has far fewer)
    uint32_t *T = calloc((size_t)bound * 8, 4);   // per id: defining opcode, three operands, location + 1, built-in + 1, flags, word index
    buf_t o = {0}, pro = {0}, *cop = NULL;
    int ncop = 0, ncop_alloc = 0;
    if (!T) { snprintf(err, errn, "out of memory"); return NULL; }
#define OPOF(id) T[(size_t)(id) * 8]
#define A1(id) T[(size_t)(id) * 8 + 1]
#define A2(id) T[(size_t)(id) * 8 + 2]
#define A3(id) T[(size_t)(id) * 8 + 3]
#define LOC(id) T[(size_t)(id) * 8 + 4]
#define BI(id) T[(size_t)(id) * 8 + 5]
#define FL(id) T[(size_t)(id) * 8 + 6]   // 1 per-primitive, 2 a chain into the module's workgroup ID (mesh), 4 a converted output, 8 the stage memory
#define AT(id) T[(size_t)(id) * 8 + 7]
#define OK(id) ((id) && (id) < bound)
    gen_t G; memset(&G, 0, sizeof G); G.nb = bound;
    uint32_t model = 0, entry = 0, nentry = 0, threads = 0, maxv = 0, maxp = 0, topo = 0, pos_struct = 0, pos_member = ~0u, t_f32 = 0;
    uint32_t pos_var = 0, idx_var = 0, payload = 0, nemit = 0, nret = 0, has_shader = 0;
    size_t first_fn = nw, first_annot = nw, first_decl = nw;
    outv vv[32], pv[32]; int nvv = 0, npv = 0;
    uint32_t lsid[3] = {0};
    uint32_t emit_cnt[8];
    // Pass 1
    { uint32_t cur_fn = 0;
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) FAIL("bad instruction at word %zu", i);
        const uint32_t *a = w + i + 1;
        if (first_decl == nw && ((op >= OpTypeVoid && op <= 39) || (op >= 41 && op <= 52) || op == OpVariable)) first_decl = i;   // types, constants, variables
        switch (op) {
        case OpCapability: if (len >= 2 && a[0] == 1) has_shader = 1; break;
        case OpEntryPoint: nentry++; model = a[0]; entry = a[1]; break;
        case OpExecutionMode:
            if (len >= 6 && a[1] == XMLocalSize) threads = a[2] * a[3] * a[4];
            else if (len >= 4 && a[1] == XMOutputVertices) maxv = a[2];
            else if (len >= 4 && a[1] == XMOutputPrimitives) maxp = a[2];
            else if (len >= 3 && (a[1] == XMOutputTriangles || a[1] == XMOutputLines || a[1] == XMOutputPoints)) topo = a[1];
            break;
        case OpExecutionModeId: if (len >= 6 && a[1] == XMLocalSizeId) { lsid[0] = a[2]; lsid[1] = a[3]; lsid[2] = a[4]; } break;   // (its constants come later)
        case OpDecorate:
            if (first_annot == nw) first_annot = i;
            if (len >= 4 && OK(a[0])) { if (a[1] == DecLocation) LOC(a[0]) = a[2] + 1; else if (a[1] == DecBuiltIn) BI(a[0]) = a[2] + 1; }
            if (len >= 3 && OK(a[0]) && a[1] == DecPerPrimitive) FL(a[0]) |= 1;
            break;
        case OpMemberDecorate:
            if (first_annot == nw) first_annot = i;
            if (len >= 5 && a[2] == DecBuiltIn && a[3] == BIPosition) { pos_struct = a[0]; pos_member = a[1]; }
            break;
        case OpGroupDecorate: case OpGroupMemberDecorate: case OpDecorateId: case OpDecorateString: case OpMemberDecorateString:
            if (first_annot == nw) first_annot = i;
            if (op == OpGroupDecorate || op == OpGroupMemberDecorate) FAIL("decoration groups");
            break;
        case OpTypeInt: if (len >= 4 && OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = a[1]; A2(a[0]) = a[2]; if (a[1] == 32 && !a[2]) G.t_u32 = a[0]; } break;
        case OpTypeFloat: if (len >= 3 && OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = a[1]; if (a[1] == 32) t_f32 = a[0]; } break;
        case OpTypeBool: if (OK(a[0])) { OPOF(a[0]) = op; G.t_bool = a[0]; } break;
        case OpTypeVector:
            if (len >= 4 && OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = a[1]; A2(a[0]) = a[2];
                if (a[1] == G.t_u32 && G.t_u32 && a[2] == 3) G.t_v3u = a[0];
                if (a[1] == G.t_u32 && G.t_u32 && a[2] == 4) G.t_v4u = a[0];
                if (a[1] == t_f32 && t_f32 && a[2] == 4) G.t_v4f = a[0]; }
            break;
        case OpTypeArray: if (len >= 4 && OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = a[1]; A2(a[0]) = a[2]; } break;
        case OpTypeRuntimeArray: if (len >= 3 && OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = a[1]; } break;
        case OpTypeStruct: if (OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = len - 2; AT(a[0]) = (uint32_t)i; } break;
        case OpTypePointer: if (len >= 4 && OK(a[0])) { OPOF(a[0]) = op; A1(a[0]) = a[1]; A2(a[0]) = a[2]; } break;
        case OpConstant: if (len >= 4 && OK(a[1])) { OPOF(a[1]) = op; A1(a[1]) = a[0]; A2(a[1]) = a[2]; } break;
        case OpFunction: if (first_fn == nw) first_fn = i; cur_fn = len >= 3 ? a[1] : 0; break;
        case OpFunctionEnd: cur_fn = 0; break;
        case OpVariable:
            if (len >= 4 && OK(a[1])) { OPOF(a[1]) = op; A1(a[1]) = a[0]; A2(a[1]) = a[2];
                if (!cur_fn && len > 4) FAIL("a global variable with an initializer"); }
            break;
        case OpAccessChain: case OpInBoundsAccessChain:
            if (len >= 4 && OK(a[1]) && OK(a[2]) && (FL(a[2]) & 2)) { FL(a[1]) |= 2; A3(a[1]) = a[0]; }   // a chain into the module's workgroup ID (marked below once it is known)
            break;
        case OpPtrAccessChain: FAIL("OpPtrAccessChain");
        case OpSetMeshOutputsEXT: break;
        case OpEmitMeshTasksEXT:
            if (cur_fn != entry || !entry) FAIL("OpEmitMeshTasksEXT outside the entry point's function");
            if (nemit == 8) FAIL("more than 8 OpEmitMeshTasksEXT");
            emit_cnt[nemit++] = a[0];
            break;
        case OpReturn: if (cur_fn && cur_fn == entry && ++nret > 64) FAIL("more than 64 OpReturn in the entry point"); break;   // (each gets its copy-out)
        }
        // The module's workgroup ID input, once declared, before any chain into it is met (variables precede the functions)
        if (op == OpVariable && len >= 4 && !cur_fn && OK(a[1]) && BI(a[1]) == BIWorkgroupId + 1 && a[2] == SCInput) { G.wid = a[1]; FL(a[1]) |= 2; }
        if (op == OpVariable && len >= 4 && !cur_fn && OK(a[1]) && BI(a[1]) == BILocalInvocationIndex + 1 && a[2] == SCInput) G.lid = a[1];
        i += len;
    } }
    if (!threads && OK(lsid[0]) && OK(lsid[1]) && OK(lsid[2]) && OPOF(lsid[0]) == OpConstant && OPOF(lsid[1]) == OpConstant && OPOF(lsid[2]) == OpConstant)
        threads = A2(lsid[0]) * A2(lsid[1]) * A2(lsid[2]);
    if (nentry != 1) FAIL("%u entry points", nentry);
    if (model != EMTask && model != EMMesh) FAIL("execution model %u is not TaskEXT or MeshEXT", model);
    if (!threads || threads > 1024) FAIL("workgroup of %u invocations", threads);
    if (!G.t_u32) FAIL("no 32-bit unsigned type");
    if (first_fn == nw) FAIL("no function");
    if (first_annot > first_decl) first_annot = first_decl;
    if (first_annot > first_fn) first_annot = first_fn;
    int task = model == EMTask;
    info->task = task; info->threads = threads;
    // The global variables: the stage memory (a word array), the outputs (mesh), workgroup memory in all
    uint64_t shared = 0;
    for (uint32_t id = 1; id < bound; id++) {
        if (OPOF(id) != OpVariable || !OK(A1(id)) || OPOF(A1(id)) != OpTypePointer) continue;
        uint32_t sc = A2(id), pt = A2(A1(id));
        // the bytes of a workgroup array of words, vectors or structs of vectors (enough for the device's limit check)
        uint64_t bytes = 0;
        if (OK(pt) && OPOF(pt) == OpTypeArray && OK(A2(pt)) && OPOF(A2(pt)) == OpConstant) {
            uint64_t el = A1(pt), n = A2(A2(pt)), eb = 4;
            if (OK(el) && OPOF(el) == OpTypeVector) eb = A2(el) == 3 ? 16 : 4ull * A2(el);
            else if (OK(el) && OPOF(el) == OpTypeStruct) eb = 16ull * A1(el);
            bytes = n * eb;
        }
        if (sc == SCWorkgroup) shared += bytes;
        if (sc == SCTaskPayload) {
            if (payload) FAIL("two task memories");
            if (!OK(pt) || OPOF(pt) != OpTypeArray || A1(pt) != G.t_u32 || !OK(A2(pt)) || OPOF(A2(pt)) != OpConstant) FAIL("a task memory that is not an array of words");
            payload = id; FL(id) |= 8; info->payload_words = A2(A2(pt)); shared += bytes;
            if (!info->payload_words || info->payload_words > 4096) FAIL("a task memory of %u words", info->payload_words);   // (16 KiB at most)
        }
        if (sc == SCOutput) {
            if (task) FAIL("an output in a task program");
            FL(id) |= 4; shared += bytes;
            if (!OK(pt) || OPOF(pt) != OpTypeArray || !OK(A2(pt)) || OPOF(A2(pt)) != OpConstant) FAIL("output %u is not an array", id);
            uint32_t el = A1(pt), n = A2(A2(pt));
            if (BI(id) == BIPrimTriangle + 1) { if (el != G.t_v3u || !G.t_v3u) FAIL("triangle indices of another type"); idx_var = id; if (maxp && n != maxp) FAIL("triangle indices for %u primitives, %u output", n, maxp); }
            else if (BI(id) == BIPrimPoint + 1 || BI(id) == BIPrimLine + 1) FAIL("a mesh program that outputs points or lines");
            else if (BI(id)) FAIL("output built-in %u", BI(id) - 1);
            else if (el == pos_struct && pos_struct) { if (pos_var) FAIL("two position outputs"); pos_var = id; if (maxv && n != maxv) FAIL("positions for %u vertices, %u output", n, maxv); }
            else if (LOC(id) && el == G.t_v4f && G.t_v4f) {
                if (LOC(id) > 32) FAIL("output location %u", LOC(id) - 1);
                if (nvv == 32 || npv == 32) FAIL("more than 32 generic outputs");
                if (FL(id) & 1) { if (n != maxp) FAIL("per-primitive output of %u elements, %u primitives", n, maxp); pv[npv++] = (outv){id, LOC(id) - 1}; }
                else { if (n != maxv) FAIL("per-vertex output of %u elements, %u vertices", n, maxv); vv[nvv++] = (outv){id, LOC(id) - 1}; }
            } else FAIL("output %u is not the position, a vec4 generic or the triangle indices", id);
        }
    }
    if (task && !payload) FAIL("a task program without task memory");
    if (!task) {
        if (topo != XMOutputTriangles) FAIL("a mesh program whose primitives are not triangles");
        if (!maxv || !maxp || maxv > 256 || maxp > 256) FAIL("a mesh program of %u vertices / %u primitives", maxv, maxp);
        if (!pos_var || !idx_var) FAIL("a mesh program without its %s output", !pos_var ? "position" : "triangle index");
        if (!G.t_v4f) FAIL("no vec4 type");
        if (!OK(pos_struct) || pos_member >= A1(pos_struct) || w[AT(pos_struct) + 2 + pos_member] != G.t_v4f) FAIL("a position that is not a vec4");
        qsort(vv, (size_t)nvv, sizeof *vv, by_loc); qsort(pv, (size_t)npv, sizeof *pv, by_loc);
        info->max_v = maxv; info->max_p = maxp; info->nv = (uint32_t)nvv; info->np = (uint32_t)npv;
        for (int k = 0; k < nvv; k++) info->vloc[k] = vv[k].loc;
        for (int k = 0; k < npv; k++) info->ploc[k] = pv[k].loc;
        info->slot_vec4 = maxv * (1 + (uint32_t)nvv) + maxp * (1 + (uint32_t)npv);
        if (!nret) FAIL("an entry point without OpReturn");
    } else if (!nemit) FAIL("a task program without OpEmitMeshTasksEXT");
    if (shared > (1u << 20)) FAIL("%llu bytes of workgroup memory", (unsigned long long)shared);
    info->shared_bytes = (uint32_t)shared;
    uint32_t pw = info->payload_words;
    // New types, pointers, variables
    if (!G.t_bool) G.t_bool = G.mk_bool = G.nb++;
    if (!G.t_v3u) G.t_v3u = G.mk_v3u = G.nb++;
    if (!task && !G.t_v4u) G.t_v4u = G.mk_v4u = G.nb++;
    G.p_sb_u32 = G.nb++; G.p_wg_u32 = G.nb++; G.rta_u = G.nb++; G.st_rec = G.nb++; G.p_sb_rec = G.nb++; G.v_rec = G.nb++;
    if (!task) { G.p_sb_v4u = G.nb++; G.p_wg_v4f = G.nb++; G.p_wg_v3u = G.nb++; G.rta_v4u = G.nb++; G.st_out = G.nb++; G.p_sb_out = G.nb++; G.v_out = G.nb++; G.p_priv_v3u = G.nb++; }
    if (!task || !G.wid) { G.p_in_v3u = G.nb++; G.wid_new = G.nb++; }   // mesh: the module's own input becomes a private copy of the index its body sees
    if (!G.lid) { G.p_in_u32 = G.nb++; G.lid_new = G.nb++; G.lid = G.lid_new; }
    uint32_t wid_rd = task && G.wid ? G.wid : G.wid_new;
    // Private pointers for chains into the module's workgroup ID (their pointee types, in order of first use)
    uint32_t chain_pt[8], chain_new[8]; int nch = 0;
    if (!task) for (uint32_t id = 1; id < bound; id++) if ((FL(id) & 2) && id != G.wid) {
        uint32_t rt = A3(id), pt = OK(rt) && OPOF(rt) == OpTypePointer ? A2(rt) : 0; int k = 0;
        if (!pt) FAIL("a chain into the workgroup ID of an unknown type");
        while (k < nch && chain_pt[k] != pt) k++;
        if (k == nch) { if (nch == 8) FAIL("too many chain types into the workgroup ID"); chain_pt[nch] = pt; chain_new[nch++] = G.nb++; }
    }
    // The code: a mesh program's prologue and output copies, a task's record copies (the constants they use are made as they go)
    if (!task) {
        uint32_t lidv = G.nb++, wr = G.nb++, gi = G.nb++;
        I(&pro, OpLoad, G.t_u32, lidv, G.lid); I(&pro, OpLoad, G.t_v3u, wr, wid_rd); I(&pro, OpCompositeExtract, G.t_u32, gi, wr, 0);
        uint32_t S = rec_word(&pro, &G, cu(&G, VKA_ME_S)), mode = rec_word(&pro, &G, cu(&G, VKA_ME_MODE)), mb = rec_word(&pro, &G, cu(&G, VKA_ME_MAPBASE));
        uint32_t base = rec_word(&pro, &G, cu(&G, VKA_ME_BASE)), gb = G.nb++;   // this workgroup's mesh workgroup: the round's first one + its index in the dispatch
        I(&pro, OpIAdd, G.t_u32, gb, gi, base);
        uint32_t ist = G.nb++, mi = G.nb++; I(&pro, OpINotEqual, G.t_bool, ist, mode, cu(&G, 0)); I(&pro, OpIAdd, G.t_u32, mi, mb, gb);
        uint32_t e = rec_word(&pro, &G, mi), t0 = G.nb++, m0 = G.nb++, t = G.nb++, m = G.nb++, wv = G.nb++;
        I(&pro, OpBitwiseAnd, G.t_u32, t0, e, cu(&G, 0xffff)); I(&pro, OpShiftRightLogical, G.t_u32, m0, e, cu(&G, 16));
        I(&pro, OpSelect, G.t_u32, t, ist, t0, cu(&G, 0)); I(&pro, OpSelect, G.t_u32, m, ist, m0, gb);
        I(&pro, OpCompositeConstruct, G.t_v3u, wv, m, cu(&G, 0), cu(&G, 0));
        if (G.wid) I(&pro, OpStore, G.wid, wv);   // the body's workgroup index: its own among those its task launched
        if (payload) {   // the task memory from the task's record (words past the record read 0)
            uint32_t tS = G.nb++, rb = G.nb++, sm1 = G.nb++;
            I(&pro, OpIMul, G.t_u32, tS, t, S); I(&pro, OpIAdd, G.t_u32, rb, tS, cu(&G, VKA_ME_REC0 + 1)); I(&pro, OpISub, G.t_u32, sm1, S, cu(&G, 1));
            for (uint32_t k = 0; k * threads < pw; k++) {
                uint32_t id = elem(&pro, &G, lidv, k, threads, pw), inr = G.nb++, ok = G.nb++, src = G.nb++, val = G.nb++, pd = G.nb++;
                I(&pro, OpULessThan, G.t_bool, inr, id, sm1); I(&pro, OpLogicalAnd, G.t_bool, ok, ist, inr); I(&pro, OpIAdd, G.t_u32, src, rb, id);
                uint32_t v = rec_word(&pro, &G, src);
                I(&pro, OpSelect, G.t_u32, val, ok, v, cu(&G, 0)); I(&pro, OpAccessChain, G.p_wg_u32, pd, payload, id); I(&pro, OpStore, pd, val);
            }
            barrier(&pro, &G);
        }
        // the output copy before each return of the entry point (the wrapper has one)
        if (!(cop = calloc(nret, sizeof *cop))) FAIL("out of memory");
        ncop_alloc = (int)nret;
        uint32_t nV1 = 1 + (uint32_t)nvv, nP1 = 1 + (uint32_t)npv, prim0 = maxv * nV1;
        for (ncop = 0; ncop < (int)nret; ncop++) {
            buf_t *b = &cop[ncop];
            barrier(b, &G);
            uint32_t lv = G.nb++, wr2 = G.nb++, g2 = G.nb++, sb = G.nb++;
            I(b, OpLoad, G.t_u32, lv, G.lid); I(b, OpLoad, G.t_v3u, wr2, wid_rd); I(b, OpCompositeExtract, G.t_u32, g2, wr2, 0); I(b, OpIMul, G.t_u32, sb, g2, cu(&G, info->slot_vec4));
            for (uint32_t k = 0; k * threads < maxv; k++) {
                uint32_t v = elem(b, &G, lv, k, threads, maxv), vm = G.nb++, vb = G.nb++;
                I(b, OpIMul, G.t_u32, vm, v, cu(&G, nV1)); I(b, OpIAdd, G.t_u32, vb, sb, vm);
                for (int j = -1; j < nvv; j++) {
                    uint32_t pp = G.nb++, x = G.nb++, xu = G.nb++, dst = vb, po = G.nb++;
                    if (j < 0) I(b, OpAccessChain, G.p_wg_v4f, pp, pos_var, v, cu(&G, pos_member)); else I(b, OpAccessChain, G.p_wg_v4f, pp, vv[j].var, v);
                    I(b, OpLoad, G.t_v4f, x, pp); I(b, OpBitcast, G.t_v4u, xu, x);
                    if (j >= 0) { dst = G.nb++; I(b, OpIAdd, G.t_u32, dst, vb, cu(&G, 1 + (uint32_t)j)); }
                    I(b, OpAccessChain, G.p_sb_v4u, po, G.v_out, cu(&G, 0), dst); I(b, OpStore, po, xu);
                }
            }
            for (uint32_t k = 0; k * threads < maxp; k++) {
                uint32_t t = elem(b, &G, lv, k, threads, maxp), tm = G.nb++, pb0 = G.nb++, pb = G.nb++;
                I(b, OpIMul, G.t_u32, tm, t, cu(&G, nP1)); I(b, OpIAdd, G.t_u32, pb0, sb, cu(&G, prim0)); I(b, OpIAdd, G.t_u32, pb, pb0, tm);
                uint32_t ip = G.nb++, iv = G.nb++, i0 = G.nb++, i1 = G.nb++, i2 = G.nb++, i4 = G.nb++, po = G.nb++;
                I(b, OpAccessChain, G.p_wg_v3u, ip, idx_var, t); I(b, OpLoad, G.t_v3u, iv, ip);
                I(b, OpCompositeExtract, G.t_u32, i0, iv, 0); I(b, OpCompositeExtract, G.t_u32, i1, iv, 1); I(b, OpCompositeExtract, G.t_u32, i2, iv, 2);
                I(b, OpCompositeConstruct, G.t_v4u, i4, i0, i1, i2, cu(&G, 0)); I(b, OpAccessChain, G.p_sb_v4u, po, G.v_out, cu(&G, 0), pb); I(b, OpStore, po, i4);
                for (int j = 0; j < npv; j++) {
                    uint32_t pp = G.nb++, x = G.nb++, xu = G.nb++, dst = G.nb++, po2 = G.nb++;
                    I(b, OpAccessChain, G.p_wg_v4f, pp, pv[j].var, t); I(b, OpLoad, G.t_v4f, x, pp); I(b, OpBitcast, G.t_v4u, xu, x);
                    I(b, OpIAdd, G.t_u32, dst, pb, cu(&G, 1 + (uint32_t)j)); I(b, OpAccessChain, G.p_sb_v4u, po2, G.v_out, cu(&G, 0), dst); I(b, OpStore, po2, xu);
                }
            }
        }
    } else {
        // each OpEmitMeshTasksEXT: the count (at most VKA_ME_MAXM) and the task memory into this workgroup's record, then the end; each OpReturn of the entry point (a
        // workgroup that ends without launching any): a count of 0, so the record is not the one an earlier draw left
        if (!(cop = calloc(nemit + nret, sizeof *cop))) FAIL("out of memory");
        ncop_alloc = (int)(nemit + nret);
        for (ncop = (int)nemit; ncop < (int)(nemit + nret); ncop++) {
            buf_t *b = &cop[ncop];
            uint32_t wr = G.nb++, t = G.nb++, tS = G.nb++, rb = G.nb++, pr = G.nb++;
            I(b, OpLoad, G.t_v3u, wr, wid_rd); I(b, OpCompositeExtract, G.t_u32, t, wr, 0);
            uint32_t S = rec_word(b, &G, cu(&G, VKA_ME_S));
            I(b, OpIMul, G.t_u32, tS, t, S); I(b, OpIAdd, G.t_u32, rb, tS, cu(&G, VKA_ME_REC0));
            I(b, OpAccessChain, G.p_sb_u32, pr, G.v_rec, cu(&G, 0), rb); I(b, OpStore, pr, cu(&G, 0));
        }
        for (ncop = 0; ncop < (int)nemit; ncop++) {
            buf_t *b = &cop[ncop];
            uint32_t lv = G.nb++, wr = G.nb++, t = G.nb++;
            barrier(b, &G);   // every invocation reaches it (it is executed in uniform control flow); the task memory its others wrote is read below
            I(b, OpLoad, G.t_u32, lv, G.lid); I(b, OpLoad, G.t_v3u, wr, wid_rd); I(b, OpCompositeExtract, G.t_u32, t, wr, 0);
            uint32_t S = rec_word(b, &G, cu(&G, VKA_ME_S)), mm = rec_word(b, &G, cu(&G, VKA_ME_MAXM)), lt = G.nb++, cc = G.nb++, tS = G.nb++, rb = G.nb++, pr = G.nb++, rb1 = G.nb++;
            I(b, OpULessThan, G.t_bool, lt, emit_cnt[ncop], mm); I(b, OpSelect, G.t_u32, cc, lt, emit_cnt[ncop], mm);
            I(b, OpIMul, G.t_u32, tS, t, S); I(b, OpIAdd, G.t_u32, rb, tS, cu(&G, VKA_ME_REC0));
            I(b, OpAccessChain, G.p_sb_u32, pr, G.v_rec, cu(&G, 0), rb); I(b, OpStore, pr, cc); I(b, OpIAdd, G.t_u32, rb1, rb, cu(&G, 1));
            for (uint32_t k = 0; k * threads < pw; k++) {
                uint32_t id = elem(b, &G, lv, k, threads, pw), pp = G.nb++, x = G.nb++, dst = G.nb++, pd = G.nb++;
                I(b, OpAccessChain, G.p_wg_u32, pp, payload, id); I(b, OpLoad, G.t_u32, x, pp); I(b, OpIAdd, G.t_u32, dst, rb1, id);
                I(b, OpAccessChain, G.p_sb_u32, pd, G.v_rec, cu(&G, 0), dst); I(b, OpStore, pd, x);
            }
            binst(b, OpReturn, NULL, 0);
        }
        ncop = (int)(nemit + nret);
    }
    G.full = 1;
    if (pro.bad) FAIL("out of memory");
    for (int k = 0; k < ncop; k++) if (cop[k].bad) FAIL("out of memory");
    if (G.over) FAIL("too many constants");
    // Pass 2: the module
    bputn(&o, w, 5);
    { uint32_t cur_fn = 0; int pro_st = 0, kret = 0, kemit = 0;   // pro_st: 0 before the entry block, 1 after its label (its variables may follow), 2 placed
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i, *a = in + 1;
        if (i == first_annot) {   // the new decorations, before the module's
            I(&o, OpDecorate, G.rta_u, DecArrayStride, 4); I(&o, OpDecorate, G.st_rec, DecBlock); I(&o, OpMemberDecorate, G.st_rec, 0, DecOffset, 0);
            I(&o, OpDecorate, G.v_rec, DecDescriptorSet, 0); I(&o, OpDecorate, G.v_rec, DecBinding, rec_binding);
            if (!task) { I(&o, OpDecorate, G.rta_v4u, DecArrayStride, 16); I(&o, OpDecorate, G.st_out, DecBlock); I(&o, OpMemberDecorate, G.st_out, 0, DecOffset, 0);
                         I(&o, OpDecorate, G.v_out, DecDescriptorSet, 0); I(&o, OpDecorate, G.v_out, DecBinding, out_binding); }
            if (G.wid_new) I(&o, OpDecorate, G.wid_new, DecBuiltIn, BIWorkgroupId);
            if (G.lid_new) I(&o, OpDecorate, G.lid_new, DecBuiltIn, BILocalInvocationIndex);
        }
        if (i == first_fn) {   // the new types, constants and variables, before the functions
            if (G.mk_bool) I(&o, OpTypeBool, G.mk_bool);
            if (G.mk_v3u) I(&o, OpTypeVector, G.mk_v3u, G.t_u32, 3);
            if (G.mk_v4u) I(&o, OpTypeVector, G.mk_v4u, G.t_u32, 4);
            I(&o, OpTypePointer, G.p_sb_u32, SCStorageBuffer, G.t_u32); I(&o, OpTypePointer, G.p_wg_u32, SCWorkgroup, G.t_u32);
            I(&o, OpTypeRuntimeArray, G.rta_u, G.t_u32); I(&o, OpTypeStruct, G.st_rec, G.rta_u); I(&o, OpTypePointer, G.p_sb_rec, SCStorageBuffer, G.st_rec);
            I(&o, OpVariable, G.p_sb_rec, G.v_rec, SCStorageBuffer);
            if (!task) {
                I(&o, OpTypePointer, G.p_sb_v4u, SCStorageBuffer, G.t_v4u); I(&o, OpTypePointer, G.p_wg_v4f, SCWorkgroup, G.t_v4f); I(&o, OpTypePointer, G.p_wg_v3u, SCWorkgroup, G.t_v3u);
                I(&o, OpTypeRuntimeArray, G.rta_v4u, G.t_v4u); I(&o, OpTypeStruct, G.st_out, G.rta_v4u); I(&o, OpTypePointer, G.p_sb_out, SCStorageBuffer, G.st_out);
                I(&o, OpVariable, G.p_sb_out, G.v_out, SCStorageBuffer);
                I(&o, OpTypePointer, G.p_priv_v3u, SCPrivate, G.t_v3u);
                if (G.wid) I(&o, OpVariable, G.p_priv_v3u, G.wid, SCPrivate);   // the module's workgroup ID, now the index the prologue gives the body
                for (int k = 0; k < nch; k++) I(&o, OpTypePointer, chain_new[k], SCPrivate, chain_pt[k]);
            }
            if (G.wid_new) { I(&o, OpTypePointer, G.p_in_v3u, SCInput, G.t_v3u); I(&o, OpVariable, G.p_in_v3u, G.wid_new, SCInput); }
            if (G.lid_new) { I(&o, OpTypePointer, G.p_in_u32, SCInput, G.t_u32); I(&o, OpVariable, G.p_in_u32, G.lid_new, SCInput); }
            for (int k = 0; k < G.nc; k++) I(&o, OpConstant, G.t_u32, G.cid[k], G.cval[k]);
        }
        if (op == OpFunction) cur_fn = len >= 3 ? a[1] : 0;
        if (pro_st == 1 && op != OpVariable) { bputn(&o, pro.w, pro.n); pro_st = 2; }
        switch (op) {
        case OpCapability: if (len >= 2 && a[0] == CapMeshShading) { if (!has_shader) { I(&o, OpCapability, 1); has_shader = 1; } goto skip; } break;   // (MeshShadingEXT implies Shader)
        case OpExtension: if (str_is(a, len - 1, "SPV_EXT_mesh_shader")) goto skip; break;
        case OpEntryPoint: {
            uint32_t sw = str_words(a + 2, len - 3), n0 = 3 + sw;   // model, function, name, then the interface
            buf_t e = {0};
            bput(&e, EMGLCompute); bput(&e, a[1]); bputn(&e, a + 2, sw);
            for (uint32_t k = n0; k < len; k++) bput(&e, in[k]);
            uint32_t add[5] = {G.v_rec, G.v_out, G.wid_new, G.lid_new, 0};
            for (int k = 0; k < 4; k++) if (add[k] && (ver >= 0x00010400 || add[k] == G.wid_new || add[k] == G.lid_new)) bput(&e, add[k]);
            if (e.bad) { free(e.w); FAIL("out of memory"); }
            binst(&o, OpEntryPoint, e.w, e.n); free(e.w);
            goto skip; }
        case OpExecutionMode: if (len >= 3 && (a[1] == XMOutputVertices || a[1] == XMOutputPrimitives || a[1] == XMOutputTriangles)) goto skip; break;
        case OpDecorate: case OpDecorateId: case OpDecorateString:
            if (len >= 2 && OK(a[0]) && (FL(a[0]) & 4)) goto skip;   // an output's location, built-in, per-primitive: workgroup memory has none
            if (len >= 3 && op == OpDecorate && OK(a[0]) && a[0] == pos_struct && a[1] == DecBlock) goto skip;
            if (len >= 3 && op == OpDecorate && !task && a[0] == G.wid && a[1] == DecBuiltIn) goto skip;
            break;
        case OpMemberDecorate: case OpMemberDecorateString: if (len >= 2 && a[0] == pos_struct && pos_struct) goto skip; break;
        case OpTypePointer: if (len >= 4 && (a[1] == SCOutput || a[1] == SCTaskPayload)) { I(&o, OpTypePointer, a[0], SCWorkgroup, a[2]); goto skip; } break;
        case OpVariable:
            if (!cur_fn && len >= 4 && (a[2] == SCOutput || a[2] == SCTaskPayload)) { I(&o, OpVariable, a[0], a[1], SCWorkgroup); goto skip; }
            if (!cur_fn && !task && a[1] == G.wid && G.wid) goto skip;   // re-emitted as a private variable before the functions
            break;
        case OpAccessChain: case OpInBoundsAccessChain:
            if (!task && len >= 4 && OK(a[1]) && (FL(a[1]) & 2)) {
                uint32_t pt = A2(a[0]); int k = 0; while (k < nch && chain_pt[k] != pt) k++;
                bputn(&o, in, len); o.w[o.n - len + 1] = chain_new[k]; goto skip; }
            break;
        case OpLabel: bputn(&o, in, len); if (cur_fn == entry && !task && !pro_st) pro_st = 1; goto skip;
        case OpSetMeshOutputsEXT: goto skip;
        case OpEmitMeshTasksEXT: bputn(&o, cop[kemit].w, cop[kemit].n); kemit++; goto skip;
        case OpReturn: if (cur_fn == entry) { const buf_t *c = &cop[task ? nemit + (uint32_t)kret : (uint32_t)kret]; bputn(&o, c->w, c->n); kret++; } break;
        case OpFunctionEnd: cur_fn = 0; break;
        }
        bputn(&o, in, len);
skip:
        i += len;
    } }
    if (o.bad) FAIL("out of memory");
    o.w[3] = G.nb;
    for (int k = 0; k < ncop; k++) free(cop[k].w);
    free(cop); free(pro.w); free(T);
    *out_nw = o.n; return o.w;
fail:
    for (int k = 0; k < ncop_alloc; k++) free(cop[k].w);
    free(cop); free(pro.w); free(o.w); free(T);
    return NULL;
#undef FAIL
#undef OPOF
#undef A1
#undef A2
#undef A3
#undef LOC
#undef BI
#undef FL
#undef AT
#undef OK
}

// ---------- vka_spv_mesh_vs ----------
uint32_t *vka_spv_mesh_vs(const vka_spv_mesh_info *in, uint32_t out_binding, size_t *out_nw) {
    if (in->task || !in->max_v || !in->max_p || in->nv > 32 || in->np > 32) return NULL;
    buf_t o = {0};
    uint32_t nb = 1;
    uint32_t t_void = nb++, t_fn = nb++, t_u32 = nb++, t_i32 = nb++, t_f32 = nb++, t_v4f = nb++, t_v4u = nb++, t_bool = nb++, rta = nb++, st = nb++, p_sb = nb++, buf = nb++,
             p_sb_v4u = nb++, p_in_i32 = nb++, vid = nb++, p_out_v4f = nb++, pos = nb++, main_fn = nb++;
    uint32_t ov[32], op_[32];
    for (uint32_t j = 0; j < in->nv; j++) ov[j] = nb++;
    for (uint32_t j = 0; j < in->np; j++) op_[j] = nb++;
    uint32_t vps = 3 * in->max_p, nV1 = 1 + in->nv, nP1 = 1 + in->np, prim0 = in->max_v * nV1;
    // constants: 0, 1, 3, vps, slot, nV1, nP1, prim0, max_v, max_v - 1, then 1 + j
    enum { NCK = 10 };
    uint32_t ck[NCK + 33], cv[NCK + 33]; int nck = 0;
    #define CK(v) ({ uint32_t v_ = (v), r_ = 0; for (int q_ = 0; q_ < nck; q_++) if (cv[q_] == v_) r_ = ck[q_]; if (!r_) { cv[nck] = v_; r_ = ck[nck++] = nb++; } r_; })
    uint32_t c0 = CK(0), c1 = CK(1), c3 = CK(3), cvps = CK(vps), csv = CK(in->slot_vec4), cnv = CK(nV1), cnp = CK(nP1), cp0 = CK(prim0), cmv = CK(in->max_v), cmv1 = CK(in->max_v - 1);
    uint32_t cj[33]; for (uint32_t j = 0; j < 33; j++) cj[j] = j + 1 <= (in->nv > in->np ? in->nv : in->np) ? CK(1 + j) : 0;
    bput(&o, 0x07230203); bput(&o, 0x00010300); bput(&o, 0); bput(&o, 0); bput(&o, 0);
    I(&o, OpCapability, 1);   // Shader
    I(&o, OpMemoryModel, 0, 1);   // Logical GLSL450
    { buf_t e = {0}; bput(&e, EMVertex); bput(&e, main_fn); bput(&e, 0x6e69616d); bput(&e, 0); bput(&e, vid); bput(&e, pos);
      for (uint32_t j = 0; j < in->nv; j++) bput(&e, ov[j]);
      for (uint32_t j = 0; j < in->np; j++) bput(&e, op_[j]);
      binst(&o, OpEntryPoint, e.w, e.n); free(e.w); }
    I(&o, OpDecorate, vid, DecBuiltIn, BIVertexIndex); I(&o, OpDecorate, pos, DecBuiltIn, BIPosition);
    for (uint32_t j = 0; j < in->nv; j++) I(&o, OpDecorate, ov[j], DecLocation, in->vloc[j]);
    for (uint32_t j = 0; j < in->np; j++) { I(&o, OpDecorate, op_[j], DecLocation, in->ploc[j]); I(&o, OpDecorate, op_[j], DecFlat); }
    I(&o, OpDecorate, rta, DecArrayStride, 16); I(&o, OpDecorate, st, DecBlock); I(&o, OpMemberDecorate, st, 0, DecOffset, 0); I(&o, OpMemberDecorate, st, 0, DecNonWritable);
    I(&o, OpDecorate, buf, DecDescriptorSet, 0); I(&o, OpDecorate, buf, DecBinding, out_binding);
    I(&o, OpTypeVoid, t_void); I(&o, OpTypeFunction, t_fn, t_void); I(&o, OpTypeInt, t_u32, 32, 0); I(&o, OpTypeInt, t_i32, 32, 1); I(&o, OpTypeFloat, t_f32, 32);
    I(&o, OpTypeVector, t_v4f, t_f32, 4); I(&o, OpTypeVector, t_v4u, t_u32, 4); I(&o, OpTypeBool, t_bool);
    I(&o, OpTypeRuntimeArray, rta, t_v4u); I(&o, OpTypeStruct, st, rta); I(&o, OpTypePointer, p_sb, SCStorageBuffer, st); I(&o, OpTypePointer, p_sb_v4u, SCStorageBuffer, t_v4u);
    I(&o, OpTypePointer, p_in_i32, SCInput, t_i32); I(&o, OpTypePointer, p_out_v4f, SCOutput, t_v4f);
    for (int q = 0; q < nck; q++) I(&o, OpConstant, t_u32, ck[q], cv[q]);
    I(&o, OpVariable, p_sb, buf, SCStorageBuffer); I(&o, OpVariable, p_in_i32, vid, SCInput); I(&o, OpVariable, p_out_v4f, pos, SCOutput);
    for (uint32_t j = 0; j < in->nv; j++) I(&o, OpVariable, p_out_v4f, ov[j], SCOutput);
    for (uint32_t j = 0; j < in->np; j++) I(&o, OpVariable, p_out_v4f, op_[j], SCOutput);
    I(&o, OpFunction, t_void, main_fn, 0, t_fn); I(&o, OpLabel, nb++);
    uint32_t vi = nb++, v = nb++, slot = nb++, r = nb++, t = nb++, c = nb++, sb = nb++, tm = nb++, pb0 = nb++, pb = nb++;
    I(&o, OpLoad, t_i32, vi, vid); I(&o, OpBitcast, t_u32, v, vi);
    I(&o, OpUDiv, t_u32, slot, v, cvps); I(&o, OpUMod, t_u32, r, v, cvps); I(&o, OpUDiv, t_u32, t, r, c3); I(&o, OpUMod, t_u32, c, r, c3);
    I(&o, OpIMul, t_u32, sb, slot, csv); I(&o, OpIMul, t_u32, tm, t, cnp); I(&o, OpIAdd, t_u32, pb0, sb, cp0); I(&o, OpIAdd, t_u32, pb, pb0, tm);
    uint32_t ip = nb++, iu = nb++, i0 = nb++, i1 = nb++, i2 = nb++, is0 = nb++, is1 = nb++, s12 = nb++, vx = nb++, inr = nb++, vc = nb++, vm = nb++, vb = nb++;
    I(&o, OpAccessChain, p_sb_v4u, ip, buf, c0, pb); I(&o, OpLoad, t_v4u, iu, ip);
    I(&o, OpCompositeExtract, t_u32, i0, iu, 0); I(&o, OpCompositeExtract, t_u32, i1, iu, 1); I(&o, OpCompositeExtract, t_u32, i2, iu, 2);
    I(&o, OpIEqual, t_bool, is0, c, c0); I(&o, OpIEqual, t_bool, is1, c, c1); I(&o, OpSelect, t_u32, s12, is1, i1, i2); I(&o, OpSelect, t_u32, vx, is0, i0, s12);
    I(&o, OpULessThan, t_bool, inr, vx, cmv); I(&o, OpSelect, t_u32, vc, inr, vx, cmv1); I(&o, OpIMul, t_u32, vm, vc, cnv); I(&o, OpIAdd, t_u32, vb, sb, vm);
    for (int j = -1; j < (int)in->nv + (int)in->np; j++) {
        int prim = j >= (int)in->nv; uint32_t at = j < 0 ? vb : nb++, p = nb++, x = nb++, f = nb++;
        if (j >= 0) I(&o, OpIAdd, t_u32, at, prim ? pb : vb, cj[prim ? j - in->nv : (uint32_t)j]);
        I(&o, OpAccessChain, p_sb_v4u, p, buf, c0, at); I(&o, OpLoad, t_v4u, x, p); I(&o, OpBitcast, t_v4f, f, x);
        I(&o, OpStore, j < 0 ? pos : prim ? op_[j - in->nv] : ov[j], f);
    }
    binst(&o, OpReturn, NULL, 0); binst(&o, 56, NULL, 0);   // OpFunctionEnd
    #undef CK
    if (o.bad) { free(o.w); return NULL; }
    o.w[3] = nb;
    *out_nw = o.n; return o.w;
}

// ---------- vka_spv_fs_mesh_flat ----------
uint32_t *vka_spv_fs_mesh_flat(const uint32_t *w, size_t nw, size_t *out_nw, int *changed, char *err, size_t errn) {
    *changed = 0; err[0] = 0;
    if (nw < 5 || w[0] != 0x07230203) { snprintf(err, errn, "not SPIR-V"); return NULL; }
    uint32_t bound = w[3];
    if (bound > (1u << 22)) { snprintf(err, errn, "an id bound of %u", bound); return NULL; }
    uint8_t *flat = calloc(bound, 1);   // bit 0: the id has Flat; bits 1-7: its first members with Flat (member m: bit m + 1, m < 7)
    int has_shader = 0;
    if (!flat) { snprintf(err, errn, "out of memory"); return NULL; }
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > nw) { free(flat); snprintf(err, errn, "bad instruction at word %zu", i); return NULL; }
        if (op == OpCapability && len >= 2 && w[i + 1] == 1) has_shader = 1;
        if (op == OpDecorate && len >= 3 && w[i + 1] < bound && w[i + 2] == DecFlat) flat[w[i + 1]] |= 1;
        if (op == OpMemberDecorate && len >= 4 && w[i + 1] < bound && w[i + 3] == DecFlat && w[i + 2] < 7) flat[w[i + 1]] |= (uint8_t)(2u << w[i + 2]);
        if ((op == OpDecorate && len >= 3 && w[i + 2] == DecPerPrimitive) || (op == OpMemberDecorate && len >= 4 && w[i + 3] == DecPerPrimitive)) *changed = 1;
        i += len;
    }
    if (!*changed) { free(flat); return NULL; }
    buf_t o = {0};
    bputn(&o, w, 5);
    for (size_t i = 5; i < nw;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16; const uint32_t *in = w + i;
        if (op == OpCapability && len >= 2 && in[1] == CapMeshShading) { if (!has_shader) { I(&o, OpCapability, 1); has_shader = 1; } i += len; continue; }
        if (op == OpExtension && str_is(in + 1, len - 1, "SPV_EXT_mesh_shader")) { i += len; continue; }
        if (op == OpDecorate && len >= 3 && in[2] == DecPerPrimitive) { if (in[1] >= bound || !(flat[in[1]] & 1)) I(&o, OpDecorate, in[1], DecFlat); i += len; continue; }
        if (op == OpMemberDecorate && len >= 4 && in[3] == DecPerPrimitive) {
            if (in[1] >= bound || in[2] >= 7 || !(flat[in[1]] >> (in[2] + 1) & 1)) I(&o, OpMemberDecorate, in[1], in[2], DecFlat);
            i += len; continue; }
        bputn(&o, in, len);
        i += len;
    }
    free(flat);
    if (o.bad) { free(o.w); snprintf(err, errn, "out of memory"); return NULL; }
    *out_nw = o.n; return o.w;
}
