/* Source-level interaction checks. This is not browser/visual QA. */
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
const root=new URL('./dist/',import.meta.url);
const nodes=new Map();
function node(id){if(!nodes.has(id))nodes.set(id,{innerHTML:'',textContent:'',value:'',dataset:{},focus(){},style:{},setAttribute(k,v){this[k]=v},getBoundingClientRect(){return {left:0,bottom:0,width:200,height:30}}});return nodes.get(id);}
const source=fs.readFileSync(new URL('app.js',root),'utf8');
const context=vm.createContext({window:{scrollTo(){}},document:{getElementById:node,querySelectorAll(){return[]},querySelector(){return node('query')},addEventListener(){}},location:{hash:''},innerWidth:1440,innerHeight:1000,console});
context.window.addEventListener=()=>{};
vm.runInContext(fs.readFileSync(new URL('assets/results.js',root),'utf8'),context);
vm.runInContext(source,context);
function run(s){return vm.runInContext(s,context);}
function valid(label){const html=node('main').innerHTML;assert(html.length>2000,label);assert(!/NaN|undefined|Infinity/.test(html),`${label} includes invalid output`);assert(!/cx="-|cy="-/.test(html),`${label} chart marker outside view`);return html;}
for(const page of ['overview','data','inference','prediction','validation','reproduction','improvement']){
 run(`state.page='${page}';render()`);valid(page);
}
for(const group of ['grade','fico','income','purpose','interest']){run(`state.eda='${group}';state.page='data';render()`);valid(group);}
for(const scenario of ['primary','mature','unwinsorized','employment_complete_case'])for(const model of ['A','B','C']){
 run(`state.scenario='${scenario}';state.inferenceModel='${model}';state.page='inference';render()`);valid(scenario+model);
}
for(const curve of ['roc','pr']){run(`state.curve='${curve}';state.page='prediction';render()`);valid(curve);}
const models=run('D.metrics.map(r=>r.model)');
for(const model of models){run(`state.model=${JSON.stringify(model)};state.page='validation';render()`);valid(model);}
for(let rule=0;rule<5;rule++){run(`state.rule=${rule};state.page='validation';render()`);valid('rule'+rule);}
for(const group of run('[...new Set(D.subgroups.map(r=>r.subgroup_variable))]')){run(`state.subgroup=${JSON.stringify(group)};state.page='validation';render()`);valid(group);}
run("state.page='validation';render()");node('threshold-range').value='0';node('threshold-range').oninput();const before=node('threshold-summary').innerHTML;node('threshold-range').value='158';node('threshold-range').oninput();assert.notEqual(before,node('threshold-summary').innerHTML);assert.equal(node('threshold-value').textContent,'0.800');
run("state.page='reproduction';render()");node('check-search').value='zzzznone';node('check-search').oninput();assert(node('checks-table').innerHTML.includes('No matching checks'));node('check-search').value='calibration';node('check-search').oninput();assert(node('checks-table').innerHTML.includes('calibration'));
for(const rule of ['maximum_validation_f1','validation_cost_5_to_1']){run(`state.page='improvement';state.improvementRule='${rule}';render()`);valid(rule);}
for(const curve of ['roc','pr']){run(`state.page='improvement';state.improvementCurve='${curve}';render()`);valid(curve);}
const D=run('D');assert.equal(D.cleaning.actual.final_row_count,1348099);assert(D.audit.complete);assert.equal(D.audit.passed,58);assert.equal(D.thresholds[0].tp+D.thresholds[0].fn,8874);
assert(D.xgboost);assert.equal(D.xgboost.development.length,6);assert.equal(D.xgboost.comparison.length,2);for(const file of Object.values(D.xgboost.sources))assert(fs.existsSync(new URL('assets/tables/'+file,root)));
assert.equal(D.improvement.candidate,'expanded_grade_31');assert(D.improvement.comparison[1].pr_auc>D.improvement.comparison[0].pr_auc);for(const file of Object.values(D.improvement.sources))assert(fs.existsSync(new URL('assets/tables/'+file,root)));
for(const file of Object.values(D.sources))assert(fs.existsSync(new URL('assets/tables/'+file,root)));
assert(!/"id"\s*:|"predicted_probability"\s*:/.test(fs.readFileSync(new URL('assets/results.json',root),'utf8')),'Individual loan data exported');
console.log('Passed: 7 pages; exploratory improvement rules/curves; EDA, inference and model filters; subgroup and rule selectors; slider; search; downloads; source totals; aggregate-only export.');
