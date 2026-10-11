// C ABI over the unmodified MiaCode parser, ordered-area judge and serializers.
// No application, window, event loop, audio player, renderer or child process.
#define main maichart_unused_muri_dump_main
#include "upstream/src/tools/muri/MuriDump.cpp"
#undef main
#include "upstream/src/tools/muri/MuriSlideReferenceData.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#ifdef _WIN32
#define MURI_EXPORT extern "C" __declspec(dllexport)
#else
#define MURI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

MURI_EXPORT int maichart_muri_abi() { return 1; }
MURI_EXPORT void maichart_muri_free(void* ptr) { std::free(ptr); }
MURI_EXPORT char* maichart_muri_analyze(const char* utf8, double firstSeconds) {
  QByteArray output;
  try {
    if (!utf8 || !qIsFinite(firstSeconds)) throw std::runtime_error("invalid input");
    if (miacode::muri::detail::slideRuntimeRoot().isEmpty())
      throw std::runtime_error("missing Slide reference data");
    const QString chartText = QString::fromUtf8(utf8);
    const QString sourceDescription = QStringLiteral("maichart-memory");
    const QString difficultyName = QStringLiteral("selected-inote");
    const SimaiNativeParseResult nativeResult = SimaiNativeParser::parseForTimeline(chartText);
    const QVector<TimelineNoteMarker> shiftedMarkers =
        shiftedNoteMarkers(nativeResult.noteMarkers, firstSeconds, NonFiniteHandling::PassThrough);
    const MuriAnalysisReport report = MuriAnalyzer::analyze(shiftedMarkers);
    QJsonObject parseObject;
    parseObject.insert(QStringLiteral("ok"), nativeResult.ok);
    parseObject.insert(QStringLiteral("error_count"), nativeResult.errors.size());
    parseObject.insert(QStringLiteral("warning_count"), nativeResult.warnings.size());

    QJsonArray parseErrors;
    for (const SimaiNativeMessage& error : nativeResult.errors) {
        parseErrors.append(jsonFromParseMessage(error));
    }
    parseObject.insert(QStringLiteral("errors"), parseErrors);

    QJsonArray parseWarnings;
    for (const SimaiNativeMessage& warning : nativeResult.warnings) {
        parseWarnings.append(jsonFromParseMessage(warning));
    }
    parseObject.insert(QStringLiteral("warnings"), parseWarnings);

    QJsonObject countsByKind;
    int earlyClearedCount = 0;
    for (const MuriDiagnostic& diagnostic : report.diagnostics) {
        const QString kind = muriKindKey(diagnostic.kind);
        countsByKind.insert(kind, countsByKind.value(kind).toInt() + 1);
    }

    QList<MarkerMuriState> markerStates = report.markerStates.values();
    std::sort(markerStates.begin(), markerStates.end(), [](const MarkerMuriState& a, const MarkerMuriState& b) {
        if (!qFuzzyCompare(a.second + 1.0, b.second + 1.0)) {
            return a.second < b.second;
        }
        if (a.line != b.line) {
            return a.line < b.line;
        }
        if (a.col != b.col) {
            return a.col < b.col;
        }
        return a.markerKey < b.markerKey;
    });
    for (const MarkerMuriState& state : markerStates) {
        if (state.earlyCleared) {
            ++earlyClearedCount;
        }
    }

    const TimelineChartCounts chartCounts = countTimelineChartNotes(shiftedMarkers);
    const QVector<MuriStaticReference> staticReferences = miacode::muri::buildStaticMuriReferences(
        shiftedMarkers,
        static_cast<double>(miacode::muri::kStaticTapOnSlideThresholdDefaultMs) / 1000.0);
    QJsonObject staticReferenceCountsByKind;
    for (const MuriStaticReference& record : staticReferences) {
        const QString kind = muriKindKey(record.kind);
        staticReferenceCountsByKind.insert(kind, staticReferenceCountsByKind.value(kind).toInt() + 1);
    }

    QJsonObject summaryObject;
    summaryObject.insert(QStringLiteral("note_count"), chartCounts.noteCount);
    summaryObject.insert(QStringLiteral("marker_count"), chartCounts.markerCount);
    summaryObject.insert(QStringLiteral("tap_count"), chartCounts.tapCount);
    summaryObject.insert(QStringLiteral("hold_count"), chartCounts.holdCount);
    summaryObject.insert(QStringLiteral("slide_count"), chartCounts.slideCount);
    summaryObject.insert(QStringLiteral("touch_count"), chartCounts.touchCount);
    summaryObject.insert(QStringLiteral("break_count"), chartCounts.breakCount);
    summaryObject.insert(QStringLiteral("helper_head_tap_count"), chartCounts.helperHeadTapCount);
    summaryObject.insert(QStringLiteral("helper_head_break_count"), chartCounts.helperHeadBreakCount);
    summaryObject.insert(QStringLiteral("diagnostic_count"), report.diagnostics.size());
    summaryObject.insert(QStringLiteral("marker_state_count"), report.markerStates.size());
    summaryObject.insert(QStringLiteral("early_cleared_count"), earlyClearedCount);
    summaryObject.insert(QStringLiteral("counts_by_kind"), countsByKind);
    summaryObject.insert(QStringLiteral("static_reference_count"), staticReferences.size());
    summaryObject.insert(QStringLiteral("static_reference_counts_by_kind"), staticReferenceCountsByKind);

    QJsonArray diagnosticsArray;
    for (const MuriDiagnostic& diagnostic : report.diagnostics) {
        diagnosticsArray.append(jsonFromDiagnostic(diagnostic));
    }

    QJsonArray staticReferencesArray;
    for (const MuriStaticReference& record : staticReferences) {
        staticReferencesArray.append(jsonFromStaticReferenceRecord(record));
    }

    QJsonArray markerStatesArray;
    for (const MarkerMuriState& state : markerStates) {
        markerStatesArray.append(jsonFromMarkerState(state));
    }

    const QVector<miacode::muri::MuriPanelEntry> visibleEntries =
        miacode::muri::buildVisibleMuriPanelEntries(report, staticReferences);
    QJsonArray visibleEntriesArray;
    for (const miacode::muri::MuriPanelEntry& entry : visibleEntries) {
        QJsonObject obj;
        obj.insert(QStringLiteral("kind"), muriKindKey(entry.kind));
        obj.insert(QStringLiteral("is_static"), entry.isStatic);
        obj.insert(QStringLiteral("line"), entry.line);
        obj.insert(QStringLiteral("col"), entry.col);
        obj.insert(QStringLiteral("second"), entry.second);
        obj.insert(QStringLiteral("occurrence_second"), entry.occurrenceSecond);
        obj.insert(QStringLiteral("detail_kind"), muriDetailKindKey(entry.detailKind));
        obj.insert(QStringLiteral("detail_args"), jsonFromMuriDetailArgs(entry.detailArgs));
        obj.insert(QStringLiteral("detail"), entry.rawDetail);
        visibleEntriesArray.append(obj);
    }

    QJsonObject root;
    root.insert(QStringLiteral("ok"), nativeResult.ok);
    root.insert(QStringLiteral("source"), sourceDescription);
    root.insert(QStringLiteral("difficulty"), difficultyName);
    root.insert(QStringLiteral("first"), firstSeconds);
    root.insert(QStringLiteral("parse"), parseObject);
    root.insert(QStringLiteral("summary"), summaryObject);
    root.insert(QStringLiteral("diagnostics"), diagnosticsArray);
    root.insert(QStringLiteral("static_references"), staticReferencesArray);
    root.insert(QStringLiteral("visible_panel_entries"), visibleEntriesArray);
    root.insert(QStringLiteral("marker_states"), markerStatesArray);
    QJsonArray markersArray;
    for (const TimelineNoteMarker& marker : shiftedMarkers) markersArray.append(jsonFromMarker(marker));
    root.insert(QStringLiteral("markers"), markersArray);


    output = QJsonDocument(root).toJson(QJsonDocument::Compact);
  } catch (const std::exception& e) {
    output = QJsonDocument(QJsonObject{{"ok", false}, {"bridge_error", QString::fromUtf8(e.what())}}).toJson();
  } catch (...) {
    output = "{\"ok\":false,\"bridge_error\":\"native exception\"}";
  }
  char* result = static_cast<char*>(std::malloc(output.size()+1));
  if (result) std::memcpy(result, output.constData(), output.size()+1);
  return result;
}
