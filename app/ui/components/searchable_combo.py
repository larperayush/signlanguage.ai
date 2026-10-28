"""
Searchable Camera Dropdown Component.
Provides filterable device selection across physical and virtual capture devices.
"""
from typing import List, Optional
from PySide6.QtWidgets import QComboBox, QLineEdit, QCompleter
from PySide6.QtCore import Qt, Signal

from app.core.camera_manager import DeviceInfo
from app.resources.icons import get_icon


class SearchableCameraComboBox(QComboBox):
    """
    Searchable combobox supporting arbitrary virtual & hardware camera devices.
    Displays user-friendly descriptions while storing unique device IDs.
    """
    device_chosen = Signal(str)  # Emits device id_str

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.lineEdit().setPlaceholderText("Select or search camera...")
        
        # Configure case-insensitive auto-completion
        self.completer().setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer().setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer().setCompletionMode(QCompleter.CompletionMode.PopupCompletion)

        self._devices: List[DeviceInfo] = []
        self._is_updating = False

        self.currentIndexChanged.connect(self._on_index_changed)

    def populate_devices(self, devices: List[DeviceInfo], selected_id: Optional[str] = None):
        """Update items with newly discovered camera hardware."""
        self._is_updating = True
        self.clear()
        self._devices = devices

        if not devices:
            self.addItem("No cameras detected", userData="")
            self.setEnabled(False)
            self._is_updating = False
            return

        self.setEnabled(True)
        target_index = 0

        for idx, dev in enumerate(devices):
            label = dev.description
            if dev.is_default:
                label += " (Default)"
            self.addItem(label, userData=dev.id_str)

            if selected_id and dev.id_str == selected_id:
                target_index = idx

        self.setCurrentIndex(target_index)
        self._is_updating = False

    def _on_index_changed(self, index: int):
        if self._is_updating or index < 0:
            return
        dev_id = self.itemData(index)
        if dev_id:
            self.device_chosen.emit(dev_id)

    def get_selected_id(self) -> Optional[str]:
        return self.currentData()
