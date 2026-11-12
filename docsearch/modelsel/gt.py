"""Hand-read ground truth: text lines read by eye from zoomed crops of the two real drawings.

(doc, category, text). Single visual lines, written as printed; the scorer normalises width / x vs × / φ vs ø etc.
`critical` marks strings the three benchmark questions depend on.
"""
SYS = "sys"    # C.D棟 汙水處理系統平面圖、剖面圖
RAFT = "raft"  # C.D棟 汙水排水設備 筏基層聯通管平面圖

GT = [
    # ---- equipment legend table (sys) : dense table, mixed CJK + units
    (SYS, "legend", 'φ60cm防臭式鑄鐵蓋', 0), (SYS, "legend", 'φ75cm防臭式鑄鐵蓋', 0),
    (SYS, "legend", '浮渣擋板(FRP/SUS304)', 0), (SYS, "legend", '整流設備(FRP/SUS304)', 0),
    (SYS, "legend", '溢流堰+擋板(FRP/SUS304)', 0), (SYS, "legend", '接觸曝氣池細氣泡散氣盤', 0),
    (SYS, "legend", '調整池粗氣泡散氣盤', 0), (SYS, "legend", '消毒池粗氣泡散氣盤', 0),
    (SYS, "legend", '接觸曝氣池繩狀濾材(濾材φ50mm,比表面積≧5m2/m)', 0),
    (SYS, "legend", '調整池泵浦(0.40m3/min×8mH×2HP)', 1), (SYS, "legend", '終沉池泵浦(0.25m3/min×7mH×1HP)', 1),
    (SYS, "legend", '放流池泵浦(0.4m3/min×18mH×5HP)(屬水電工程)', 1),
    (SYS, "legend", '調整池立式魯氏鼓風機(φ3.0",10HP,5.42CMM,P=4000mmAq)', 1),
    (SYS, "legend", '接觸曝氣池立式魯氏鼓風機(φ4.0",15HP,7.31CMM,P=5000mmAq)', 1),
    (SYS, "legend", '除臭抽風機(風量≧10.0CMM,1.0HP,風壓≧200mmAq)', 1),
    (SYS, "legend", 'NaOCl加藥機×2台,PE貯藥桶1000L×1只', 0), (SYS, "legend", '螺旋式流量計(4")', 0),
    # ---- calculation sheet (sys) : dense paragraphs of thin CAD font with formulas
    (SYS, "calc", '設計水量:Q=425CMD', 0), (SYS, "calc", '設計水力停留時間=12hr,則所需容積:', 1),
    (SYS, "calc", 'V=425m3/日×12÷24≒212.5m3', 1), (SYS, "calc", '(1)L790×W352.5×H350/D320', 1),
    (SYS, "calc", '(3)L467.5×W300×H350/D320', 1), (SYS, "calc", '有效容積:223.12m3>212.5m3----OK!', 1),
    (SYS, "calc", '停留時間:223.12÷410×24≒13.1(hr)', 0), (SYS, "calc", '需要池體體積:68/0.3=226.67m3', 0),
    (SYS, "calc", '設BOD容積負荷=0.3kgBOD/m3.d(依據技術規範)', 0),
    (SYS, "calc", '採用池體有效容積:280.33m3>226.67m3....OK', 0), (SYS, "calc", 'DT=280.33÷410×24≒16.4(hr)', 0),
    (SYS, "calc", 'L=68÷280.33≒0.24kg/m3.d<0.3kg/m3.d.....OK', 0),
    (SYS, "calc", '總需氧量=2.0kgO2/kgBOD5×68kg/d=136kgO2/d', 0),
    (SYS, "calc", '曝氣供氣量=136÷0.23÷1.29÷0.08÷1440≒3.98m3/min', 0),
    (SYS, "calc", '設置鼓風機風量:7.31Nm3/min(p=4000mmAq,100MM,15HP)', 1),
    (SYS, "calc", '有效容積:237.37m3>212.5m3----OK!', 0), (SYS, "calc", 'DT=237.37÷425×24≒13.4(hr)', 0),
    (SYS, "calc", '採用0.40m3/min×8.0mH×2HP×二台(交互運轉)', 1),
    (SYS, "calc", '設置鼓風出風量:5.42Nm3/min(p=4000mmAq,80MM,10HP)', 1),
    (SYS, "calc", '本單元計採用58只粗氣泡散氣盤,散氣盤出風量', 0), (SYS, "calc", '(1)L790×W467.5×H350/D315', 0),
    (SYS, "calc", 'a.設計尺寸:L790×W352.5×H350/D305', 0), (SYS, "calc", 'V=84.93×0.70≒59.45m3', 0),
    (SYS, "calc", 'DT=59.45÷425×24≒3.4(hr)...OK!', 0), (SYS, "calc", '設計表面積=27.85m2', 0),
    (SYS, "calc", '實際表面積負荷=425÷27.85=15.26m3/m2.d...OK', 0), (SYS, "calc", '設計堰長=7.9m', 0),
    (SYS, "calc", '實際堰負荷=425÷7.9≒53.8m3/m.d.....OK', 0),
    (SYS, "calc", '污泥體積=200×425÷1000÷0.02÷1030≒4.13m3/day', 0),
    (SYS, "calc", '採用0.25m3/min×7mH×1HP×2台(定時交替運轉)', 1),
    (SYS, "calc", 'BOD:200mg/l', 0), (SYS, "calc", 'COD:400mg/l', 0), (SYS, "calc", '大腸菌類:10000000CFU/100ml', 0),
    (SYS, "calc", '估計建築物最大日污水量:350.25m3/day×1.2≒420.3m3/day', 0),
    # ---- drawing labels (sys): sparse short labels scattered between lines
    (SYS, "label", '初沉池(1-1)', 1), (SYS, "label", '初沉池(1-2)', 1), (SYS, "label", '初沉池(1-3)', 1),
    (SYS, "label", '初沉池(1-4)', 1), (SYS, "label", '初沉池(1-5)', 1), (SYS, "label", '初沉池(1-6)', 1),
    (SYS, "label", '抽水池(1-7)', 0), (SYS, "label", 'H=1.85m', 1), (SYS, "label", 'H=3.5m', 1),
    (SYS, "label", '調整池(2-1)', 0), (SYS, "label", '調整池(2-2)', 0), (SYS, "label", '調整池(2-3)', 0),
    (SYS, "label", '接觸曝氣池(3-1)', 0), (SYS, "label", '接觸曝氣池(3-2)', 0), (SYS, "label", '接觸曝氣池(3-3)', 0),
    (SYS, "label", '終沉池', 0), (SYS, "label", '消毒池(5)', 0), (SYS, "label", '放流池(6)', 0), (SYS, "label", '污泥貯留池', 0),
    (SYS, "label", '6"VP接排風機φ6"TO RF', 0), (SYS, "label", '6"CIP TO RF', 0), (SYS, "label", 'TO污水處理槽(2)', 0),
    (SYS, "label", '污水處理槽(1)平面圖', 0), (SYS, "label", '7.9x3.52x1.85x2+4.675x7.9x1.85x2=239.5m3', 1),
    (SYS, "label", '污水處理(2)流程平面圖', 0), (SYS, "label", '7.7x3.52x3.5x6+4.675x7.9x3.5x4=379m3', 1),
    (SYS, "label", '1.鼓風機至污水處理池曝氣主管採用SUS304 sch20,', 0),
    (SYS, "label", '2.曝氣管:處理池內部曝氣主管、支管、分支管採用PVC(B)管.', 0),
    (SYS, "label", '5.進出管通氣立管屬水電工程.', 0), (SYS, "label", '新增RC隔牆屬營造工程', 0),
    # ---- raft-slab drawing (raft): pump data tables printed in ~8 px text
    (RAFT, "pump-table", '滯洪放流泵規格', 1), (RAFT, "pump-table", '交替運轉', 0), (RAFT, "pump-table", '>1860', 1),
    (RAFT, "pump-table", 'CT-611-150', 1), (RAFT, "pump-table", '廠牌:川源', 0),
    (RAFT, "pump-table", '污水泵C規格', 1), (RAFT, "pump-table", '污水泵A規格', 1), (RAFT, "pump-table", '交替並列運轉', 0),
    (RAFT, "pump-table", '>500', 1), (RAFT, "pump-table", '前置雨水泵規格', 1), (RAFT, "pump-table", '單獨運轉', 0),
    (RAFT, "pump-table", '>267', 1),
    # ---- raft-slab drawing: tiny coloured pipe / tank labels over a dense grid of lines
    (RAFT, "pipe-label", '4"PVC,連通管x1支(壁上方)', 1), (RAFT, "pipe-label", '6"PVC,連通管x2支', 1),
    (RAFT, "pipe-label", 'B.O.P+320', 1), (RAFT, "pipe-label", 'B.O.P+175', 0), (RAFT, "pipe-label", 'B.O.P+20', 0),
    (RAFT, "pipe-label", '初沉池(1-1)', 1), (RAFT, "pipe-label", '初沉池(1-2)', 1), (RAFT, "pipe-label", '初沉池(1-3)', 1),
    (RAFT, "pipe-label", '調整池(2-1)', 1), (RAFT, "pipe-label", '污泥貯留池', 0),
    (RAFT, "pipe-label", '污水泵A', 0), (RAFT, "pipe-label", '5HPx2', 0), (RAFT, "pipe-label", '污水泵B', 0), (RAFT, "pipe-label", '1HPx2', 0),
    (RAFT, "pipe-label", 'φ3"SUS304(鼓風機)', 0), (RAFT, "pipe-label", 'φ4"SUS304(鼓風機)', 0),
    (RAFT, "pipe-label", 'φ6"VP(排風機)', 0), (RAFT, "pipe-label", 'φ4"SWP(B1F)', 0), (RAFT, "pipe-label", 'φ8"SP', 0),
    (RAFT, "pipe-label", '滯洪池26', 0), (RAFT, "pipe-label", '滯洪池35', 0), (RAFT, "pipe-label", '60X60cm', 0),
]
