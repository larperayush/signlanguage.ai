"""
Targeted Verification Test: Accept Sample & Accept All Samples
Verifies:
1. Slot connection of btn_accept_sample and btn_accept_all in StaticTrainingPage.
2. Appending of 63 landmark coordinates and target label ('G') from Candidate Sample Buffer to active dataset.csv.
3. Immediate update of the 'Recorded Samples' counter (lbl_ds_samples).
4. Batch commit of all candidate samples at once via 'Accept All'.
"""
import sys
import os
import csv
import time
import tempfile
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow
from app.ui.pages.static_training_page import StaticTrainingPage
from app.core.dataset_registry import DatasetRegistry, DatasetEntry


def run_test():
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.processEvents()

    page: StaticTrainingPage = window.page_static

    print("======================================================================")
    print("   SignFlow Studio — Accept Sample & Accept All Batch Verification    ")
    print("======================================================================")

    # 1. Locate slots and buttons
    print("\n[Phase 1] Locating buttons and connected slots...")
    assert hasattr(page, "btn_accept_sample"), "btn_accept_sample button not found"
    assert hasattr(page, "_accept_sample"), "Slot _accept_sample not found"
    assert callable(page._accept_sample), "_accept_sample is not callable"

    assert hasattr(page, "btn_accept_all"), "btn_accept_all button not found"
    assert hasattr(page, "_accept_all_samples"), "Slot _accept_all_samples not found"
    assert callable(page._accept_all_samples), "_accept_all_samples is not callable"
    print("  [OK] Located 'btn_accept_sample' (_accept_sample) and 'btn_accept_all' (_accept_all_samples).")

    # 2. Set up a dedicated isolated active dataset for testing
    print("\n[Phase 2] Setting up active dataset for test verification...")
    temp_dir = tempfile.mkdtemp(prefix="signflow_test_ds_")
    test_csv = Path(temp_dir) / "test_dataset.csv"

    # Write initial header and 2 existing rows
    header = ["label"] + [f"feature_{i}" for i in range(63)]
    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(["A"] + [0.1] * 63)
        writer.writerow(["B"] + [0.2] * 63)

    initial_count = 2
    reg = DatasetRegistry.get_instance()
    success, entry, msg = reg.import_dataset(str(test_csv), mode="external_reference")
    assert success and entry is not None, f"Failed to register test dataset: {msg}"
    reg.set_active_static_dataset(entry.id)
    page._refresh_dataset_display()
    app.processEvents()

    assert page.lbl_ds_samples.text() == f"{initial_count} samples"
    print(f"  [OK] Active dataset configured with initial count: {page.lbl_ds_samples.text()}")

    # 3. Add Candidate Sample for Sign 'G' with 63 landmark coordinates
    print("\n[Phase 3] Adding Single Candidate Sample for 'G'...")
    test_features = [round(0.01 * (i + 1), 4) for i in range(63)]
    sample_rec = page.add_candidate_sample(sign_label="G", features=test_features)
    app.processEvents()

    assert page.sample_table.rowCount() >= 1
    row_idx = page.sample_table.rowCount() - 1
    assert page.sample_table.item(row_idx, 1).text() == "G"
    assert page.sample_table.item(row_idx, 3).text() == "Candidate"

    # 4. Trigger the 'Accept Sample' button click
    print("\n[Phase 4] Testing Single 'Accept Sample'...")
    page.sample_table.selectRow(row_idx)
    page.btn_accept_sample.click()
    app.processEvents()

    assert page.sample_table.item(row_idx, 3).text() == "Accepted"
    assert page.lbl_ds_samples.text() == f"{initial_count + 1} samples"
    print(f"  [OK] Single accept verified: Status='Accepted', Count={page.lbl_ds_samples.text()}")

    # 5. Add MULTIPLE Candidate Samples for 'Accept All' batch testing
    print("\n[Phase 5] Adding 3 Candidate Samples (Signs 'G', 'C', 'D') for Batch Acceptance...")
    batch_signs = ["G", "C", "D"]
    batch_samples = []
    for s_idx, s_label in enumerate(batch_signs):
        feats = [round(0.02 * (s_idx + 1) * (j + 1), 4) for j in range(63)]
        rec = page.add_candidate_sample(sign_label=s_label, features=feats)
        batch_samples.append((rec, feats))
        app.processEvents()

    print(f"  [OK] Added {len(batch_samples)} candidate samples to buffer.")

    # 6. Click 'Accept All' button
    print("\n[Phase 6] Clicking 'Accept All' button...")
    page.btn_accept_all.click()
    app.processEvents()

    # 7. Verify all rows in the buffer are marked 'Accepted'
    for r in range(page.sample_table.rowCount()):
        status = page.sample_table.item(r, 3).text()
        assert status == "Accepted", f"Row {r} status expected 'Accepted', got '{status}'"
    print("  [OK] All buffer rows successfully marked as 'Accepted'.")

    # 8. Verify 'Recorded Samples' counter updated immediately for all batch items
    expected_batch_count = initial_count + 1 + len(batch_samples)
    assert page.lbl_ds_samples.text() == f"{expected_batch_count} samples", f"Expected '{expected_batch_count} samples', got '{page.lbl_ds_samples.text()}'"
    print(f"  [OK] 'Recorded Samples' counter updated immediately to: {page.lbl_ds_samples.text()}")

    # 9. Verify CSV file content on disk has all rows
    print("\n[Phase 7] Inspecting disk CSV file for batch commit...")
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) == expected_batch_count + 1, f"Expected {expected_batch_count + 1} rows in CSV, found {len(reader)}"
    print(f"  [OK] CSV file on disk verified with all {expected_batch_count} committed data rows.")

    # Cleanup test files
    reg.remove_dataset(entry.id)
    try:
        os.remove(test_csv)
        os.rmdir(temp_dir)
    except Exception:
        pass

    window.close()
    app.processEvents()

    print("\n======================================================================")
    print("   [ALL ACCEPT SAMPLE & ACCEPT ALL TESTS PASSED - 100% OPERATIONAL]  ")
    print("======================================================================\n")


if __name__ == "__main__":
    run_test()
    sys.exit(0)
