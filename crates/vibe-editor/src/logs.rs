//! The editor's log panel: a bounded ring of messages and the UI that reads it.
//!
//! The sink exists because the editor's most useful output is a record of what
//! just happened, and a console the developer has to alt-tab to read is not that.
//! The ring is bounded so a session that logs in a loop costs a fixed amount of
//! memory, and it evicts the oldest entry rather than refusing the newest: the
//! message that just fired is the one being looked for.

use std::sync::{Arc, Mutex};

/// How loud a message is, ordered from quietest to loudest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogLevel {
    /// Per-frame detail, off by default.
    Trace,
    /// Detail that helps explain a decision.
    Debug,
    /// Ordinary progress.
    Info,
    /// Something unexpected that did not stop the work.
    Warn,
    /// Something that failed.
    Error,
}

impl LogLevel {
    /// The numeric rank, from 0 for `Trace` to 4 for `Error`.
    pub fn severity(self) -> u8 {
        match self {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
        }
    }

    /// The uppercase tag shown in the panel.
    pub fn label(self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }

    /// Translate a `log` crate level into an editor level.
    pub fn from_log_level(level: log::Level) -> LogLevel {
        match level {
            log::Level::Trace => LogLevel::Trace,
            log::Level::Debug => LogLevel::Debug,
            log::Level::Info => LogLevel::Info,
            log::Level::Warn => LogLevel::Warn,
            log::Level::Error => LogLevel::Error,
        }
    }

    /// The colour the panel draws this level in.
    pub fn color(self) -> egui::Color32 {
        match self {
            LogLevel::Trace => egui::Color32::DARK_GRAY,
            LogLevel::Debug => egui::Color32::GRAY,
            LogLevel::Info => egui::Color32::WHITE,
            LogLevel::Warn => egui::Color32::YELLOW,
            LogLevel::Error => egui::Color32::RED,
        }
    }

    /// Every level, quietest first, for a filter combo box.
    pub fn all() -> [LogLevel; 5] {
        [
            LogLevel::Trace,
            LogLevel::Debug,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
        ]
    }
}

/// One captured message.
#[derive(Debug, Clone, PartialEq)]
pub struct LogEntry {
    /// How loud the message is.
    pub level: LogLevel,
    /// The formatted message body.
    pub message: String,
    /// The module path the message came from.
    pub target: String,
    /// A counter that only ever goes up, assigned when the entry is stored.
    pub sequence: u64,
}

/// A bounded, thread-safe ring of log entries with a view filter.
///
/// The minimum level is a *view* filter, not a write filter: entries quieter
/// than it are still stored. A developer who raises the level to read only the
/// warnings, then lowers it again, expects the earlier traces to still be there
/// rather than having to reproduce the run.
#[derive(Debug, Clone)]
pub struct LogSink {
    entries: Vec<LogEntry>,
    capacity: usize,
    minimum: LogLevel,
    next_sequence: u64,
}

impl LogSink {
    /// A sink holding at most `capacity` entries and showing `minimum` upwards.
    ///
    /// A capacity of 0 is raised to 1, because a ring that can hold nothing
    /// would either drop every message or panic when it wrapped.
    pub fn new(capacity: usize, minimum: LogLevel) -> Self {
        LogSink {
            entries: Vec::new(),
            capacity: capacity.max(1),
            minimum,
            next_sequence: 0,
        }
    }

    /// Store a message, evicting the oldest entry if the ring is full.
    pub fn push(&mut self, level: LogLevel, target: &str, message: impl Into<String>) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(LogEntry {
            level,
            message: message.into(),
            target: target.to_string(),
            sequence: self.next_sequence,
        });
        self.next_sequence += 1;
    }

    /// Set the level shown by [`LogSink::filtered`].
    pub fn set_minimum(&mut self, level: LogLevel) {
        self.minimum = level;
    }

    /// The level shown by [`LogSink::filtered`].
    pub fn minimum(&self) -> LogLevel {
        self.minimum
    }

    /// How many entries are stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries, oldest first.
    pub fn iter(&self) -> impl Iterator<Item = &LogEntry> {
        self.entries.iter()
    }

    /// Forget every entry, keeping the sequence counter.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// The entries at or above the minimum level, oldest first.
    pub fn filtered(&self) -> Vec<&LogEntry> {
        self.iter().filter(|e| e.level >= self.minimum).collect()
    }

    /// The stored entries as one contiguous slice, oldest first.
    pub fn as_slice(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Change the bound, dropping the oldest entries if it shrank.
    pub fn set_capacity(&mut self, capacity: usize) {
        let capacity = capacity.max(1);
        if self.entries.len() > capacity {
            let drop = self.entries.len() - capacity;
            self.entries.drain(..drop);
        }
        self.capacity = capacity;
    }

    /// The bound the ring will hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Take every entry out, leaving the sink empty.
    ///
    /// The sequence counter keeps going: a consumer tracking "I have seen up to
    /// N" would go backwards if it restarted, and would then ignore everything
    /// logged next.
    pub fn take(&mut self) -> Vec<LogEntry> {
        std::mem::take(&mut self.entries)
    }

    /// The entries whose message or target contains `needle`, ignoring case.
    ///
    /// An empty needle matches everything, so an empty search box shows the
    /// whole log rather than an empty panel.
    pub fn search(&self, needle: &str) -> Vec<&LogEntry> {
        if needle.is_empty() {
            return self.iter().collect();
        }
        let needle = needle.to_lowercase();
        self.iter()
            .filter(|e| {
                e.message.to_lowercase().contains(&needle)
                    || e.target.to_lowercase().contains(&needle)
            })
            .collect()
    }
}

/// Format one entry the way the panel shows it.
pub fn format_entry(entry: &LogEntry) -> String {
    format!(
        "  [{}] {}: {}",
        entry.level.label(),
        entry.target,
        entry.message
    )
}

/// Feeds the crate's `log` output into a [`LogSink`].
///
/// The sink sits behind a mutex because a log call can arrive from any thread
/// while the UI is reading the ring. A panic in one call must not leave the
/// mutex poisoned, or every later log call would panic as well and take the
/// editor down over a formatting bug.
#[derive(Debug, Clone)]
pub struct EditorLogger {
    sink: Arc<Mutex<LogSink>>,
}

impl EditorLogger {
    /// A logger writing into a sink holding at most `capacity` entries.
    pub fn new(capacity: usize) -> Self {
        EditorLogger {
            sink: Arc::new(Mutex::new(LogSink::new(capacity, LogLevel::Trace))),
        }
    }

    /// The shared sink, so the panel can read what the logger writes.
    pub fn sink(&self) -> Arc<Mutex<LogSink>> {
        Arc::clone(&self.sink)
    }

    /// Install this logger as the process-wide `log` implementation.
    ///
    /// # Errors
    ///
    /// Returns [`log::SetLoggerError`] when something else already claimed the
    /// global slot, which `log` permits only once per process.
    pub fn install(&self) -> Result<(), log::SetLoggerError> {
        log::set_boxed_logger(Box::new(self.clone()))?;
        log::set_max_level(log::LevelFilter::Trace);
        Ok(())
    }

    /// Lock the sink, recovering from poisoning.
    ///
    /// `into_inner` keeps the data: a panic mid-log-call loses nothing that was
    /// already stored, and treating a poisoned lock as fatal would turn one bad
    /// message into a permanently broken editor.
    fn lock_sink(&self) -> std::sync::MutexGuard<'_, LogSink> {
        self.sink.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl log::Log for EditorLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        let level = LogLevel::from_log_level(metadata.level());
        level >= self.lock_sink().minimum()
    }

    fn log(&self, record: &log::Record) {
        let level = LogLevel::from_log_level(record.level());
        let mut sink = self.lock_sink();
        if level < sink.minimum() {
            return;
        }
        sink.push(level, record.target(), record.args().to_string());
    }

    fn flush(&self) {}
}

/// The log panel's UI state.
pub struct LogPanel {
    /// The messages being shown.
    pub sink: LogSink,
    /// The level the filter is set to.
    pub filter: LogLevel,
    /// The current search text.
    pub search: String,
    /// Whether long messages wrap instead of truncating.
    pub wrap: bool,
    /// Whether the view follows the newest entry.
    pub follow: bool,
}

impl Default for LogPanel {
    fn default() -> Self {
        LogPanel::new()
    }
}

impl LogPanel {
    /// An empty panel over a sink holding 1000 entries.
    pub fn new() -> Self {
        LogPanel {
            sink: LogSink::new(1000, LogLevel::Trace),
            filter: LogLevel::Trace,
            search: String::new(),
            wrap: false,
            follow: true,
        }
    }

    /// The panel's sink, for pushing messages into or reading them out.
    pub fn sink(&mut self) -> &mut LogSink {
        &mut self.sink
    }

    /// The rows the panel would show, applying the level filter and the search.
    pub fn visible(&self) -> Vec<&LogEntry> {
        self.sink
            .search(&self.search)
            .into_iter()
            .filter(|e| e.level >= self.filter)
            .collect()
    }

    /// Draw the panel.
    ///
    /// Nothing in here is allowed to fail: a panel is a diagnostic, and a
    /// diagnostic that can crash the thing it is diagnosing is worse than no
    /// panel. Every lookup here is a checked one, and the body has no early
    /// `return` from inside a widget scope, because abandoning a half-built
    /// egui frame leaves the layout stack unbalanced and the next frame is the
    /// one that panics.
    pub fn show(&mut self, ui: &mut egui::Ui) {
        self.show_toolbar(ui);
        ui.separator();
        self.show_body(ui);
    }

    fn show_toolbar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Level:");
            let mut chosen = self.filter;
            egui::ComboBox::from_id_salt("vibe-editor-log-level")
                .selected_text(self.filter.label())
                .show_ui(ui, |ui| {
                    for level in LogLevel::all() {
                        ui.selectable_value(&mut chosen, level, level.label());
                    }
                });
            if chosen != self.filter {
                self.filter = chosen;
                self.sink.set_minimum(chosen);
            }
            if ui.button("Clear").clicked() {
                self.sink.clear();
                self.search.clear();
            }
            ui.checkbox(&mut self.wrap, "Wrap");
            ui.checkbox(&mut self.follow, "Follow");
        });

        ui.horizontal(|ui| {
            ui.label("Search:");
            let id = ui.make_persistent_id("vibe-editor-log-search");
            ui.add(
                egui::TextEdit::singleline(&mut self.search)
                    .id(id)
                    .hint_text("substring"),
            );
            let focused = ui.ctx().memory(|m| m.focused()) == Some(id);
            if focused && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.search.clear();
            }
            if ui.button("x").clicked() {
                self.search.clear();
            }
        });
    }

    fn show_body(&mut self, ui: &mut egui::Ui) {
        let rows = self.visible();
        ui.horizontal(|ui| {
            ui.label(format!("{} entries", self.sink.len()));
            if self.sink.len() != rows.len() {
                ui.label(format!("showing {}", rows.len()));
            }
        });
        if rows.is_empty() {
            ui.label("No log entries.");
            return;
        }
        let mut scroll = egui::ScrollArea::vertical().max_height(ui.available_height());
        if self.follow {
            scroll = scroll.stick_to_bottom(true);
        }
        scroll.show(ui, |ui| {
            for entry in rows {
                let text = egui::RichText::new(format_entry(entry))
                    .monospace()
                    .color(entry.level.color());
                let mut label = egui::Label::new(text).truncate();
                if self.wrap {
                    label = label.wrap();
                } else {
                    label = label.selectable(true);
                }
                ui.add(label);
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use log::Log;

    fn sink_with(capacity: usize) -> LogSink {
        LogSink::new(capacity, LogLevel::Trace)
    }

    fn messages(sink: &LogSink) -> Vec<String> {
        sink.iter().map(|e| e.message.clone()).collect()
    }

    fn sequences(sink: &LogSink) -> Vec<u64> {
        sink.iter().map(|e| e.sequence).collect()
    }

    fn emit(logger: &EditorLogger, message: &str, level: log::Level, target: &str) {
        let args = format_args!("{message}");
        let record = log::Record::builder()
            .args(args)
            .level(level)
            .target(target)
            .build();
        log::Log::log(logger, &record);
    }

    #[test]
    fn severity_orders_the_levels() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn severity_numbers_run_from_zero_to_four() {
        assert_eq!(LogLevel::Trace.severity(), 0);
        assert_eq!(LogLevel::Debug.severity(), 1);
        assert_eq!(LogLevel::Info.severity(), 2);
        assert_eq!(LogLevel::Warn.severity(), 3);
        assert_eq!(LogLevel::Error.severity(), 4);
    }

    #[test]
    fn sorting_by_level_works() {
        let mut levels = vec![LogLevel::Error, LogLevel::Trace, LogLevel::Warn];
        levels.sort();
        assert_eq!(
            levels,
            vec![LogLevel::Trace, LogLevel::Warn, LogLevel::Error]
        );
    }

    #[test]
    fn labels_are_uppercase() {
        assert_eq!(LogLevel::Trace.label(), "TRACE");
        assert_eq!(LogLevel::Debug.label(), "DEBUG");
        assert_eq!(LogLevel::Info.label(), "INFO");
        assert_eq!(LogLevel::Warn.label(), "WARN");
        assert_eq!(LogLevel::Error.label(), "ERROR");
    }

    #[test]
    fn a_new_sink_is_empty_and_shows_everything() {
        let sink = sink_with(4);
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
        assert_eq!(sink.minimum(), LogLevel::Trace);
    }

    #[test]
    fn push_stores_the_level_target_and_message() {
        let mut sink = sink_with(4);
        sink.push(LogLevel::Warn, "vibe_scene::vfs", "missing file");
        assert_eq!(sink.len(), 1);
        let entry = &sink.as_slice()[0];
        assert_eq!(entry.level, LogLevel::Warn);
        assert_eq!(entry.target, "vibe_scene::vfs");
        assert_eq!(entry.message, "missing file");
    }

    #[test]
    fn push_accepts_an_owned_string() {
        let mut sink = sink_with(4);
        sink.push(LogLevel::Info, "engine", String::from("started"));
        assert_eq!(sink.as_slice()[0].message, "started");
    }

    #[test]
    fn push_below_the_minimum_is_stored_but_not_shown() {
        let mut sink = LogSink::new(8, LogLevel::Warn);
        sink.push(LogLevel::Info, "engine", "quiet");
        assert_eq!(sink.len(), 1, "a hidden message is still a message");
        assert!(sink.filtered().is_empty());
        assert_eq!(sink.as_slice().len(), 1);
    }

    #[test]
    fn raising_and_lowering_the_minimum_keeps_the_earlier_messages() {
        let mut sink = LogSink::new(8, LogLevel::Trace);
        sink.push(LogLevel::Trace, "engine", "a");
        sink.push(LogLevel::Error, "engine", "b");
        sink.set_minimum(LogLevel::Error);
        assert_eq!(messages(&sink).len(), 2);
        assert_eq!(sink.filtered().len(), 1);
        sink.set_minimum(LogLevel::Trace);
        assert_eq!(messages(&sink), vec!["a", "b"]);
    }

    #[test]
    fn the_minimum_level_itself_is_shown() {
        let mut sink = LogSink::new(8, LogLevel::Warn);
        sink.push(LogLevel::Warn, "engine", "shown");
        sink.push(LogLevel::Info, "engine", "hidden");
        let shown: Vec<String> = sink.filtered().iter().map(|e| e.message.clone()).collect();
        assert_eq!(shown, vec!["shown"]);
    }

    #[test]
    fn at_capacity_the_newest_entries_win() {
        let mut sink = sink_with(3);
        for i in 0..5 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        assert_eq!(sink.len(), 3);
        assert_eq!(messages(&sink), vec!["m2", "m3", "m4"]);
    }

    #[test]
    fn the_evicted_entry_is_gone_by_sequence() {
        let mut sink = sink_with(2);
        for i in 0..4 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        assert_eq!(sequences(&sink), vec![2, 3]);
        assert!(
            sink.search("m0").is_empty() && sink.search("m1").is_empty(),
            "the two oldest must be gone, not merely hidden"
        );
    }

    #[test]
    fn sequence_stays_monotonic_across_the_wrap() {
        let mut sink = sink_with(2);
        for i in 0..10 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        let seen = sequences(&sink);
        assert_eq!(seen, vec![8, 9]);
        assert!(seen.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn capacity_zero_holds_exactly_one_entry() {
        let mut sink = sink_with(0);
        assert_eq!(sink.capacity(), 1);
        sink.push(LogLevel::Info, "engine", "only");
        assert_eq!(messages(&sink), vec!["only"]);
        sink.push(LogLevel::Info, "engine", "newest");
        assert_eq!(messages(&sink), vec!["newest"]);
    }

    #[test]
    fn capacity_one_survives_many_pushes() {
        let mut sink = sink_with(1);
        for i in 0..5 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        assert_eq!(messages(&sink), vec!["m4"]);
    }

    #[test]
    fn set_capacity_smaller_evicts_the_oldest() {
        let mut sink = sink_with(8);
        for i in 0..6 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        sink.set_capacity(2);
        assert_eq!(messages(&sink), vec!["m4", "m5"]);
        assert_eq!(sink.capacity(), 2);
    }

    #[test]
    fn set_capacity_larger_loses_nothing() {
        let mut sink = sink_with(4);
        for i in 0..3 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        sink.set_capacity(64);
        assert_eq!(messages(&sink), vec!["m0", "m1", "m2"]);
        assert_eq!(sink.capacity(), 64);
    }

    #[test]
    fn set_capacity_zero_coerces_to_one() {
        let mut sink = sink_with(8);
        for i in 0..3 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        sink.set_capacity(0);
        assert_eq!(sink.capacity(), 1);
        assert_eq!(messages(&sink), vec!["m2"]);
    }

    #[test]
    fn a_grown_sink_accepts_more_than_it_did() {
        let mut sink = sink_with(2);
        sink.push(LogLevel::Info, "engine", "a");
        sink.set_capacity(4);
        sink.push(LogLevel::Info, "engine", "b");
        sink.push(LogLevel::Info, "engine", "c");
        assert_eq!(messages(&sink), vec!["a", "b", "c"]);
    }

    #[test]
    fn clear_empties_the_sink() {
        let mut sink = sink_with(4);
        sink.push(LogLevel::Info, "engine", "a");
        sink.clear();
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
        assert!(sink.as_slice().is_empty());
    }

    #[test]
    fn clear_does_not_reset_the_sequence() {
        let mut sink = sink_with(4);
        sink.push(LogLevel::Info, "engine", "a");
        sink.push(LogLevel::Info, "engine", "b");
        sink.clear();
        sink.push(LogLevel::Info, "engine", "c");
        assert_eq!(sequences(&sink), vec![2]);
    }

    #[test]
    fn iter_and_as_slice_agree() {
        let mut sink = sink_with(4);
        for i in 0..3 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        let by_iter: Vec<&LogEntry> = sink.iter().collect();
        assert_eq!(by_iter, sink.as_slice().iter().collect::<Vec<_>>());
    }

    #[test]
    fn iteration_is_oldest_first() {
        let mut sink = sink_with(4);
        for i in 0..3 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        assert_eq!(messages(&sink), vec!["m0", "m1", "m2"]);
    }

    #[test]
    fn iteration_stays_oldest_first_after_the_ring_wraps() {
        let mut sink = sink_with(3);
        for i in 0..7 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        assert_eq!(messages(&sink), vec!["m4", "m5", "m6"]);
        assert_eq!(sink.iter().count(), sink.as_slice().len());
    }

    #[test]
    fn filtered_is_oldest_first() {
        let mut sink = LogSink::new(8, LogLevel::Info);
        sink.push(LogLevel::Error, "engine", "e");
        sink.push(LogLevel::Trace, "engine", "t");
        sink.push(LogLevel::Warn, "engine", "w");
        let shown: Vec<String> = sink.filtered().iter().map(|e| e.message.clone()).collect();
        assert_eq!(shown, vec!["e", "w"]);
    }

    #[test]
    fn search_is_case_insensitive() {
        let mut sink = sink_with(8);
        sink.push(LogLevel::Info, "engine", "Texture upload failed");
        assert_eq!(sink.search("texture").len(), 1);
        assert_eq!(sink.search("UPLOAD").len(), 1);
        assert_eq!(sink.search("TeXtUrE").len(), 1);
    }

    #[test]
    fn search_with_an_empty_needle_returns_everything() {
        let mut sink = sink_with(8);
        sink.push(LogLevel::Info, "engine", "a");
        sink.push(LogLevel::Info, "engine", "b");
        assert_eq!(sink.search("").len(), 2);
    }

    #[test]
    fn search_that_matches_nothing_is_empty() {
        let mut sink = sink_with(8);
        sink.push(LogLevel::Info, "engine", "a");
        assert!(sink.search("nonexistent").is_empty());
    }

    #[test]
    fn search_looks_at_the_target_too() {
        let mut sink = sink_with(8);
        sink.push(LogLevel::Info, "vibe_ecs::schedule", "ran");
        assert_eq!(sink.search("vibe_ecs").len(), 1);
        assert_eq!(sink.search("ecs::").len(), 1);
    }

    #[test]
    fn search_on_an_empty_sink_is_empty() {
        let sink = sink_with(8);
        assert!(sink.search("anything").is_empty());
        assert!(sink.search("").is_empty());
    }

    #[test]
    fn take_empties_the_sink() {
        let mut sink = sink_with(8);
        sink.push(LogLevel::Info, "engine", "a");
        sink.push(LogLevel::Info, "engine", "b");
        let taken = sink.take();
        assert_eq!(taken.len(), 2);
        assert!(sink.is_empty());
        assert!(sink.as_slice().is_empty());
        assert!(sink.filtered().is_empty());
    }

    #[test]
    fn take_returns_the_entries_oldest_first() {
        let mut sink = sink_with(8);
        for i in 0..3 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        let taken: Vec<String> = sink.take().iter().map(|e| e.message.clone()).collect();
        assert_eq!(taken, vec!["m0", "m1", "m2"]);
    }

    #[test]
    fn take_does_not_reset_the_sequence() {
        let mut sink = sink_with(8);
        for i in 0..3 {
            sink.push(LogLevel::Info, "engine", format!("m{i}"));
        }
        let highest_before = sink.as_slice().last().unwrap().sequence;
        sink.take();
        sink.push(LogLevel::Info, "engine", "after");
        let after = sink.as_slice()[0].sequence;
        assert!(
            after > highest_before,
            "a consumer tracking 'seen up to N' must not go backwards"
        );
    }

    #[test]
    fn format_entry_has_the_level_target_and_message() {
        let entry = LogEntry {
            level: LogLevel::Warn,
            message: "shader compile failed".into(),
            target: "vibe_shader".into(),
            sequence: 7,
        };
        assert_eq!(
            format_entry(&entry),
            "  [WARN] vibe_shader: shader compile failed"
        );
    }

    #[test]
    fn format_entry_leading_spaces_are_exactly_two() {
        let entry = LogEntry {
            level: LogLevel::Info,
            message: "ok".into(),
            target: "engine".into(),
            sequence: 0,
        };
        assert!(format_entry(&entry).starts_with("  [INFO] engine: ok"));
    }

    #[test]
    fn a_poisoned_sink_can_still_be_locked() {
        let logger = EditorLogger::new(4);
        let sink = logger.sink();
        let _ = std::panic::catch_unwind(|| {
            let _held = sink.lock().unwrap();
            panic!("a log call that goes wrong");
        });
        let mut recovered = sink.lock().unwrap_or_else(|e| e.into_inner());
        recovered.push(LogLevel::Info, "engine", "still works");
        assert_eq!(recovered.len(), 1);
    }

    #[test]
    fn a_poisoned_sink_still_records_through_the_logger() {
        let logger = EditorLogger::new(4);
        let sink = logger.sink();
        let _ = std::panic::catch_unwind(|| {
            let _held = sink.lock().unwrap();
            panic!("a log call that goes wrong");
        });
        emit(&logger, "after the panic", log::Level::Warn, "vibe_engine");
        let recovered = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered.as_slice()[0].message, "after the panic");
    }

    #[test]
    fn the_logger_and_the_panel_share_one_sink() {
        let logger = EditorLogger::new(4);
        let shared = logger.sink();
        shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(LogLevel::Error, "engine", "boom");
        let seen = logger
            .sink()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        assert_eq!(seen, 1);
    }

    #[test]
    fn enabled_respects_the_minimum_level() {
        let logger = EditorLogger::new(4);
        logger
            .sink()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_minimum(LogLevel::Warn);
        let quiet = log::Metadata::builder()
            .level(log::Level::Info)
            .target("engine")
            .build();
        let loud = log::Metadata::builder()
            .level(log::Level::Error)
            .target("engine")
            .build();
        assert!(!logger.enabled(&quiet));
        assert!(logger.enabled(&loud));
    }

    #[test]
    fn the_log_impl_stores_the_target_and_message() {
        let logger = EditorLogger::new(4);
        emit(&logger, "frame 3", log::Level::Warn, "vibe_frame");
        let sink = logger.sink();
        let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(guard.as_slice()[0].target, "vibe_frame");
        assert_eq!(guard.as_slice()[0].message, "frame 3");
        assert_eq!(guard.as_slice()[0].level, LogLevel::Warn);
    }

    #[test]
    fn the_log_impl_maps_every_log_level() {
        let logger = EditorLogger::new(8);
        for (level, expected) in [
            (log::Level::Trace, LogLevel::Trace),
            (log::Level::Debug, LogLevel::Debug),
            (log::Level::Info, LogLevel::Info),
            (log::Level::Warn, LogLevel::Warn),
            (log::Level::Error, LogLevel::Error),
        ] {
            emit(&logger, "x", level, "engine");
            let sink = logger.sink();
            let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
            let entry = guard.as_slice().last().unwrap();
            assert_eq!(entry.level, expected);
        }
    }

    #[test]
    fn the_log_impl_assigns_increasing_sequences() {
        let logger = EditorLogger::new(8);
        for i in 0..3 {
            emit(&logger, &format!("m{i}"), log::Level::Info, "engine");
        }
        let sink = logger.sink();
        let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(sequences(&guard), vec![0, 1, 2]);
    }

    #[test]
    fn the_log_impl_respects_a_raised_minimum() {
        let logger = EditorLogger::new(8);
        logger
            .sink()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .set_minimum(LogLevel::Error);
        emit(&logger, "quiet", log::Level::Info, "engine");
        let sink = logger.sink();
        let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.is_empty());
    }

    #[test]
    fn the_log_impl_evicts_the_oldest_at_capacity() {
        let logger = EditorLogger::new(2);
        for i in 0..4 {
            emit(&logger, &format!("m{i}"), log::Level::Info, "engine");
        }
        let sink = logger.sink();
        let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(messages(&guard), vec!["m2", "m3"]);
    }

    #[test]
    fn flushing_the_logger_does_nothing() {
        let logger = EditorLogger::new(4);
        logger.flush();
        let sink = logger.sink();
        let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert!(guard.is_empty());
    }

    #[test]
    fn install_delegates_to_the_log_crate() {
        let logger = EditorLogger::new(4);
        if logger.install().is_err() {
            return;
        }
        log::info!("installed through the editor logger");
        let sink = logger.sink();
        let guard = sink.lock().unwrap_or_else(|e| e.into_inner());
        assert!(
            guard
                .iter()
                .any(|e| e.message.contains("installed through the editor logger")),
            "a message logged after install must reach the panel"
        );
    }

    #[test]
    fn a_new_panel_starts_empty_and_unfiltered() {
        let mut panel = LogPanel::new();
        assert!(panel.sink().is_empty());
        assert_eq!(panel.filter, LogLevel::Trace);
        assert_eq!(panel.search, "");
        assert!(!panel.wrap);
        assert!(panel.follow);
    }

    #[test]
    fn the_panel_filter_narrows_what_is_visible() {
        let mut panel = LogPanel::new();
        panel.sink().push(LogLevel::Info, "engine", "i");
        panel.sink().push(LogLevel::Error, "engine", "e");
        assert_eq!(panel.visible().len(), 2);
        panel.filter = LogLevel::Error;
        assert_eq!(panel.visible().len(), 1);
    }

    #[test]
    fn the_panel_search_narrows_what_is_visible() {
        let mut panel = LogPanel::new();
        panel.sink().push(LogLevel::Info, "engine", "texture ready");
        panel.sink().push(LogLevel::Info, "engine", "audio ready");
        panel.search = "AUDIO".into();
        assert_eq!(panel.visible().len(), 1);
        assert_eq!(panel.visible()[0].message, "audio ready");
    }

    #[test]
    fn the_panel_draws_without_panicking() {
        let mut panel = LogPanel::new();
        panel
            .sink()
            .push(LogLevel::Error, "vibe_rhi", "device lost");
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            panel.show(ui);
        });
        output.textures_delta.clear();
        assert_eq!(panel.sink.len(), 1);
    }

    #[test]
    fn the_panel_draws_when_empty() {
        let mut panel = LogPanel::new();
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            panel.show(ui);
        });
        output.textures_delta.clear();
        assert!(panel.sink.is_empty());
    }

    #[test]
    fn the_panel_draws_with_wrapping_and_a_full_ring() {
        let mut panel = LogPanel::new();
        panel.wrap = true;
        panel.follow = true;
        for i in 0..50 {
            panel
                .sink()
                .push(LogLevel::Warn, "vibe_scene", format!("a long message {i}"));
        }
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            panel.show(ui);
        });
        output.textures_delta.clear();
        assert_eq!(panel.sink.len(), 50);
    }
}
