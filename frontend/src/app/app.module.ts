import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { AppComponent } from './app.component';
import { NoopAnimationsModule } from '@angular/platform-browser/animations';
import {MatSlideToggleModule} from "@angular/material/slide-toggle";
import {MatButtonModule} from "@angular/material/button";
import {MatDialogModule} from "@angular/material/dialog";
import {MatIconModule} from "@angular/material/icon";
import {MatListModule} from "@angular/material/list";
import {MatMenuModule} from "@angular/material/menu";
import {MatToolbarModule} from "@angular/material/toolbar";
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { Routes, RouterModule } from '@angular/router';
import { HttpClientModule } from '@angular/common/http';
import {ExtendedModule, FlexLayoutModule, FlexModule} from '@angular/flex-layout';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import {MatInputModule} from "@angular/material/input";
import { HeaderComponent } from './Components/header/header.component';
import { ContactComponent } from './Components/contact/contact.component';
import { ApiInfoComponent } from './Components/api-info/api-info.component';
import { AboutComponent } from './Components/about/about.component';
import { UserInterfaceComponent } from './Components/user-interface/user-interface.component';
import {MatSelectModule} from "@angular/material/select";
import {MatProgressBarModule} from "@angular/material/progress-bar";
import {MatAutocompleteModule} from "@angular/material/autocomplete";
import {MatCardModule} from "@angular/material/card";
import {MatSliderModule} from "@angular/material/slider";
import {MatCheckboxModule} from "@angular/material/checkbox";
import { QuestionAskingComponent } from './Components/user-interface/question-asking/question-asking.component';
import { IsComparativeComponent } from './Components/user-interface/results-presentation/is-comparative/is-comparative.component';
import {AppStateService} from "./StateManagement/app-state.service";
import { ObjectsAndAspectsComponent } from './Components/user-interface/results-presentation/objects-and-aspects/objects-and-aspects.component';
import {MatFormFieldModule} from "@angular/material/form-field";
import { ScorePresentationComponent } from './Components/user-interface/results-presentation/score-presentation/score-presentation.component';
import { SummaryPresentationComponent } from './Components/user-interface/results-presentation/summary-presentation/summary-presentation.component';
import { ArgumentsPresentationComponent } from './Components/user-interface/results-presentation/arguments-presentation/arguments-presentation.component';
import {NgOptimizedImage} from "@angular/common";
import { SummaryFeedbackComponent } from './Components/user-interface/results-presentation/summary-presentation/summary-feedback/summary-feedback.component';

const appRoute: Routes = [
  { path: '', component: UserInterfaceComponent },
  { path: 'about', component: AboutComponent },
  { path: 'api-info', component: ApiInfoComponent },
  { path: 'contact', component: ContactComponent }
];

@NgModule({
  declarations: [
    AppComponent,
    HeaderComponent,
    ContactComponent,
    ApiInfoComponent,
    AboutComponent,
    UserInterfaceComponent,
    QuestionAskingComponent,
    IsComparativeComponent,
    ObjectsAndAspectsComponent,
    ScorePresentationComponent,
    SummaryPresentationComponent,
    ArgumentsPresentationComponent,
    SummaryFeedbackComponent,
  ],
    imports: [
        BrowserModule,
        NoopAnimationsModule,
        RouterModule.forRoot(appRoute, {useHash: false}),
        MatToolbarModule,
        FlexLayoutModule,
        MatMenuModule,
        MatIconModule,
        MatButtonModule,
        MatInputModule,
        MatSelectModule,
        FormsModule,
        MatProgressBarModule,
        MatAutocompleteModule,
        MatCardModule,
        MatSliderModule,
        MatListModule,
        MatCheckboxModule,
        ReactiveFormsModule,
        HttpClientModule,
        MatFormFieldModule,
        NgOptimizedImage,
    ],
  providers: [AppStateService],
  bootstrap: [AppComponent]
})
export class AppModule { }
