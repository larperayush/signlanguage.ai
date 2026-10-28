"""
Targeted Verification Test: Accept Sample Button & Dataset Append
Verifies:
1. Slot connection of btn_accept_sample in StaticTrainingPage.
2. Appending of 63 landmark coordinates and target label ('G') from Candidate Sample Buffer to active dataset.csv.
3. Immediate update of the 'Recorded Samples' counter (lbl_ds_samples).
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
    print("   SignFlow Studio — Accept Sample Slot & Dataset Verification        ")
    print("======================================================================")

    # 1. Locate the slot connected to the Accept Sample button
    print("\n[Phase 1] Locating slot connected to Accept Sample button...")
    assert hasattr(page, "btn_accept_sample"), "btn_accept_sample button not found"
    assert hasattr(page, "_accept_sample"), "Slot _accept_sample not found"
    assert callable(page._accept_sample), "_accept_sample is not callable"
    print("  [OK] Located slot '_accept_sample' connected to btn_accept_sample.")

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

    assert page.lbl_ds_samples.text() == f"{initial_count} samples", f"Expected '{initial_count} samples', got '{page.lbl_ds_samples.text()}'"
    print(f"  [OK] Active dataset configured with initial count: {page.lbl_ds_samples.text()}")

    # 3. Add Candidate Sample for Sign 'G' with 63 landmark coordinates
    print("\n[Phase 3] Adding Candidate Sample for 'G' with 63 landmark coordinates...")
    test_features = [round(0.01 * (i + 1), 4) for i in range(63)]
    assert len(test_features) == 63

    sample_rec = page.add_candidate_sample(sign_label="G", features=test_features)
    app.processEvents()

    assert page.sample_table.rowCount() >= 1, "Candidate sample table should have at least 1 row"
    row_idx = page.sample_table.rowCount() - 1
    assert page.sample_table.item(row_idx, 0).text() == sample_rec.id
    assert page.sample_table.item(row_idx, 1).text() == "G"
    assert page.sample_table.item(row_idx, 2).text() == "63 coords"
    assert page.sample_table.item(row_idx, 3).text() == "Candidate"
    print(f"  [OK] Candidate sample buffered in review table: ID={sample_rec.id}, Sign=G, Coords=63, Status=Candidate.")

    # 4. Trigger the 'Accept Sample' button click
    print("\n[Phase 4] Clicking 'Accept Sample' button...")
    page.sample_table.selectRow(row_idx)
    page.btn_accept_sample.click()
    app.processEvents()

    # 5. Verify the table status updated immediately
    assert page.sample_table.item(row_idx, 3).text() == "Accepted", "Status did not update to 'Accepted'"
    print("  [OK] Table item status updated to 'Accepted'.")

    # 6. Verify 'Recorded Samples' counter updated immediately
    expected_new_count = initial_count + 1
    recorded_samples_text = page.lbl_ds_samples.text()
    assert recorded_samples_text == f"{expected_new_count} samples", f"Expected '{expected_new_count} samples', got '{recorded_samples_text}'"
    print(f"  [OK] 'Recorded Samples' counter updated immediately: {recorded_samples_text}")

    # 7. Verify the actual CSV file content on disk
    print("\n[Phase 5] Inspecting saved CSV file on disk...")
    with open(test_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))

    assert len(reader) == expected_new_count + 1, f"Expected {expected_new_count + 1} lines (header + rows), got {len(reader)}"
    last_row = reader[-1]
    assert last_row[0] == "G", f"Expected last row label 'G', got '{last_row[0]}'"
    saved_coords = [float(v) for v in last_row[1:]]
    assert len(saved_coords) == 63, f"Expected 63 coordinate values, got {len(saved_coords)}"
    assert saved_coords == test_features, "Saved landmark coordinates do not match candidate features"
    print(f"  [OK] CSV file on disk verified: Label='G', Features=63 coordinates, exactly matching buffer.")

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
    print("   [ALL ACCEPT SAMPLE TESTS PASSED SUCCESSFULLY - 100% VERIFIED]     ")
    print("======================================================================\n")


if __name__ == "__main__":
    run_test()
    sys.exit(0)
