# ═══════════════════════════════════════════════════════════════════════════════
# SNIPPET 1: Adaugă în CONFIG (linia ~387)
# ═══════════════════════════════════════════════════════════════════════════════

"ask_pause_between_tiers": True,  # Întreabă copilul dacă vrea pauză între tier-uri


# ═══════════════════════════════════════════════════════════════════════════════
# SNIPPET 2: Adaugă în create_general_settings_tab() (după linia ~3450)
# ═══════════════════════════════════════════════════════════════════════════════

        # --- Grup pentru Setări Învățare ---
        learning_group = QGroupBox("📚 Setări Învățare")
        learning_layout = QFormLayout(learning_group)
        
        # Combobox pentru pauza între tier-uri
        self.pause_between_tiers_combo = QComboBox()
        self.pause_between_tiers_combo.addItems(["DA - Întreabă copilul", "NU - Continuă automat"])
        self.pause_between_tiers_combo.currentTextChanged.connect(self.on_pause_between_tiers_changed)
        learning_layout.addRow("Pauză de gândire după nivel:", self.pause_between_tiers_combo)
        
        # Explicație
        pause_info_label = QLabel("💡 Dacă alegi 'NU', Cucuvel va trece automat la următorul nivel fără să întrebe.")
        pause_info_label.setStyleSheet("font-size: 10px; color: #666; font-style: italic;")
        pause_info_label.setWordWrap(True)
        learning_layout.addWidget(pause_info_label)
        
        layout.addWidget(learning_group)


# ═══════════════════════════════════════════════════════════════════════════════
# SNIPPET 3: Funcția callback (adaugă după alte funcții on_*_changed, ~linia 2510)
# ═══════════════════════════════════════════════════════════════════════════════

    def on_pause_between_tiers_changed(self, text):
        """Callback când se schimbă setarea pentru pauza între tier-uri."""
        if "DA" in text:
            self.config["ask_pause_between_tiers"] = True
        else:
            self.config["ask_pause_between_tiers"] = False
        
        save_config(self.config)
        status = "activată" if self.config["ask_pause_between_tiers"] else "dezactivată"
        log_timestamp(f"⚙️ [CONFIG] Pauză între tier-uri {status}", "app")


# ═══════════════════════════════════════════════════════════════════════════════
# SNIPPET 4: Încarcă setarea în UI (_load_settings_into_ui, ~linia 2510)
# ═══════════════════════════════════════════════════════════════════════════════

        # Încarcă setarea pentru pauza între tier-uri
        if self.config.get("ask_pause_between_tiers", True):
            self.pause_between_tiers_combo.setCurrentText("DA - Întreabă copilul")
        else:
            self.pause_between_tiers_combo.setCurrentText("NU - Continuă automat")


# ═══════════════════════════════════════════════════════════════════════════════
# SNIPPET 5: ÎNLOCUIEȘTE logica tier terminat (linia ~4294-4305)
# ═══════════════════════════════════════════════════════════════════════════════

        if has_next_tier:
            next_tier = self.current_curriculum["tiers"][current_tier_index + 1]
            
            # Verifică setarea: să întrebe sau nu despre pauză
            ask_pause = self.config.get("ask_pause_between_tiers", True)
            
            if ask_pause:
                # COMPORTAMENT VECHI: Întreabă copilul
                completion_msg = f"[EMOTION:proud] Bravo, {self.current_student_name}! Ai terminat acest nivel! Vrei să continui cu următorul nivel: '{next_tier['tier_name']}', sau preferi să faci o pauză?"
                self.waiting_for_tier_decision = True
                self.next_tier_available = True
                self.pending_next_tier_id = next_tier["tier_id"]
                
                # Programăm TTS-ul cu întrebarea
                QTimer.singleShot(100, lambda: self._start_tts(completion_msg))
            else:
                # COMPORTAMENT NOU: Avansează direct fără să întrebe
                completion_msg = f"[EMOTION:proud] Bravo, {self.current_student_name}! Ai terminat acest nivel! Acum mergem mai departe la '{next_tier['tier_name']}'!"
                self.waiting_for_tier_decision = False
                self.next_tier_available = True
                self.pending_next_tier_id = next_tier["tier_id"]
                
                # Programăm TTS + avansarea automată
                QTimer.singleShot(100, lambda: self._start_tts(completion_msg))
                QTimer.singleShot(3000, self._advance_to_next_tier)  # Avansează după 3 secunde
        else:
            # Ultimul tier din curriculum
            completion_msg = f"[EMOTION:proud] Felicitări, {self.current_student_name}! Ai terminat toate nivelurile din acest domeniu! Ești grozav!"
            self.waiting_for_tier_decision = False
            
            # Programăm TTS-ul final
            QTimer.singleShot(100, lambda: self._start_tts(completion_msg))
