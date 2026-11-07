#include "friday/runtime/paged_kv_cache.h"
#include <iostream>
#include <cassert>
#include <vector>

using namespace friday::runtime;

void test_basic_allocation() {
    std::cout << "[TEST] 1. Temel Sayfa Tahsisati ve Imha...\n";
    PagedKvCache cache(48, 8, 128, 100); // 48 layer, 8 heads, 128 dim, 100 blocks
    assert(cache.get_free_blocks() == 100);

    ContextId ctx1 = cache.create_context();
    assert(cache.get_allocated_contexts() == 1);

    // 32 token yaz (2 blok: 16 + 16)
    for (int i = 0; i < 32; ++i) {
        cache.prepare_write_slot(ctx1, i);
    }

    const auto& table = cache.get_block_table(ctx1);
    assert(table.size() == 2);
    assert(cache.get_free_blocks() == 98);
    assert(cache.get_physical_block(table[0]).ref_count == 1);
    assert(cache.get_physical_block(table[1]).ref_count == 1);

    cache.destroy_context(ctx1);
    assert(cache.get_allocated_contexts() == 0);
    assert(cache.get_free_blocks() == 100); // Sifir sizinti!
    std::cout << "  -> [BASARILI] Temel tahsisat ve sizintisiz imha dogrulandi.\n";
}

void test_cow_fork_and_isolation() {
    std::cout << "[TEST] 2. Invariant 4: Copy-On-Write Fork Context & Izolasyon...\n";
    PagedKvCache cache(48, 8, 128, 100);

    // Parent baglama 32 token yaz (2 blok)
    ContextId parent = cache.create_context();
    for (int i = 0; i < 32; ++i) {
        cache.prepare_write_slot(parent, i);
    }
    assert(cache.get_free_blocks() == 98);

    // 3 adet cocuk baglam catallandir (Diyalektik Koro: Architect, Critic, Alternative)
    ContextId child_arch = cache.fork_context(parent);
    ContextId child_crit = cache.fork_context(parent);
    ContextId child_alt  = cache.fork_context(parent);

    assert(cache.get_allocated_contexts() == 4);
    // CATALLANMA ANINDA SIFIR BELLEK HARCAMASI (O(1) Fork):
    assert(cache.get_free_blocks() == 98); // Hic yeni blok harcanmadi!

    const auto& parent_table = cache.get_block_table(parent);
    assert(cache.get_physical_block(parent_table[0]).ref_count == 4);
    assert(cache.get_physical_block(parent_table[1]).ref_count == 4);
    std::cout << "  -> [BASARILI] O(1) Zero-Copy Fork dogrulandi (Ref Count: 4, Harcanan VRAM: 0 bayt).\n";

    // Simdi Child Architect baglamina 33. token'i yazalim (Yeni blok acilir: 2 blok paylasilir, 3. blok ozeldir)
    cache.prepare_write_slot(child_arch, 32);
    assert(cache.get_free_blocks() == 97); // 1 yeni blok tahsis edildi
    assert(cache.get_block_table(child_arch).size() == 3);
    assert(cache.get_block_table(parent).size() == 2); // Parent etkilenmedi!

    // Simdi Child Critic baglaminda mevcut 31. token uzerine yeni bir dal yazildigini varsayalim (COW Tetiklenir)
    // 31. token 1. bloktadir (ref_count = 4). Critic buraya yazarsa 1. blok onun icin kopyalanip ayrilmalidir.
    cache.prepare_write_slot(child_crit, 31);
    assert(cache.get_free_blocks() == 96); // COW icin 1 yeni blok harcandi!

    const auto& crit_table = cache.get_block_table(child_crit);
    // Critic artik ayri bir fiziksel bloktadir:
    assert(crit_table[1] != parent_table[1]);
    assert(cache.get_physical_block(crit_table[1]).ref_count == 1);
    // Parent blogunun ref_count'u 3'e dusmustur:
    assert(cache.get_physical_block(parent_table[1]).ref_count == 3);
    std::cout << "  -> [BASARILI] Copy-On-Write izolasyonu dogrulandi (Yazma aninda bagimsiz sayfa tahsisi).\n";

    // Tum contextleri sirayla yok et ve VRAM'in tam sifirlanmasini (Leak-Free) dogrula
    cache.destroy_context(child_arch);
    cache.destroy_context(child_crit);
    cache.destroy_context(child_alt);
    assert(cache.get_allocated_contexts() == 1);
    assert(cache.get_physical_block(parent_table[0]).ref_count == 1);
    assert(cache.get_physical_block(parent_table[1]).ref_count == 1);

    cache.destroy_context(parent);
    assert(cache.get_allocated_contexts() == 0);
    assert(cache.get_free_blocks() == 100); // Tum bloklar iade edildi!
    std::cout << "  -> [BASARILI] 100% Leak-Free VRAM iadesi dogrulandi.\n";
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  FRIDAY RUNTIME: TEST_COW_KV_CACHE (INVARIANT 4)\n";
    std::cout << "========================================================\n";
    try {
        test_basic_allocation();
        test_cow_fork_and_isolation();
        std::cout << "\n>>> [TUM TESTLER BASARIYLA TAMAMLANDI - %100 GECTI] <<<\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[TEST HATASI] " << e.what() << "\n";
        return 1;
    }
}
